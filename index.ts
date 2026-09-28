/**
 * image-compact — make images usable by models that cannot see them.
 *
 * Some models are text-only. `opencode/nemotron-3-ultra-free` is one: given a
 * PNG it replies "the current model does not support image processing". pi does
 * not fail in that case, it quietly drops the picture and leaves a placeholder,
 * so the image is simply gone and the model has nothing to work from.
 *
 * This extension replaces the picture with a text description, produced by a
 * vision-capable model, before the image can be discarded.
 *
 * ## Why the obvious hook does not work
 *
 * `before_provider_request` is the natural place to rewrite a request, and it
 * is too late. pi's provider layer calls `downgradeUnsupportedImages` inside
 * `transformMessages`, which swaps every image block for the string
 * "(tool image omitted: model n)" *before* that hook is emitted. By the time an
 * extension sees the payload the base64 is gone, so there is nothing left to
 * describe. Anything built on that hook is a no-op.
 *
 * So the conversion happens earlier, where the bytes still exist:
 *
 *   - `tool_result` — the read tool returning a PNG. The image is replaced in
 *     the result, so it never enters history and is never downgraded.
 *   - `context` — an image the user pasted or attached, which arrives as a user
 *     message rather than a tool result. `context` runs before every LLM call
 *     with the transcript intact.
 *
 * ## The trigger, and a trap
 *
 * The model's own capability, read from `ctx.model.input` at the moment of the
 * call, so providers registered at runtime (multi-account) are handled and no
 * catalog needs parsing.
 *
 * The obvious move is to keep a list of models whose catalog entry is wrong.
 * Do not. `opencode/muse-spark-1.3-contributor-free` is catalogued
 * `input: ["text"]`, yet it reads images correctly — three runs in a row, no
 * extension loaded. It also reports no image support on other runs of the very
 * same command, and pi then strips the image. A free tier that load-balances
 * across backends will not agree with itself about its own modalities.
 *
 * That makes the flag unreliable as a *description* of the model but entirely
 * reliable as a description of *this request*: when it says the model cannot
 * take images, pi is about to drop the image, so converting is strictly better
 * than losing it. Exempting a model therefore does not upgrade it, it just
 * guarantees the image disappears on the runs where the flag is pessimistic.
 * So there is no exemption list, and the runtime flag is the whole decision.
 *
 * ## Cost
 *
 * The description comes from a child `pi` on a vision model, which reuses the
 * credentials pi already holds — no API key is read, stored or forwarded here.
 * Results are cached by content hash, so a screenshot costs one nested call ever
 * rather than one per turn.
 */
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { extname, join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const AGENT_DIR = process.env.PI_CODING_AGENT_DIR?.trim() || join(homedir(), ".pi", "agent");
const CACHE_DIR = join(AGENT_DIR, "cache", "image-compact");

/** A model that can actually see, so the description is worth having. */
const VISION_MODEL =
	process.env.PI_IMAGE_COMPACT_MODEL?.trim() || "opencode-account1/muse-spark-1.3-contributor-free";
const TIMEOUT_MS = Number(process.env.PI_IMAGE_COMPACT_TIMEOUT_MS ?? 120_000);
/** Force conversion even when the model claims to accept images. */
const FORCE = process.env.PI_IMAGE_COMPACT_FORCE === "1";

/**
 * Asked of the vision model. The reader has never seen the screen, so
 * "describe the image" invites the one thing it cannot use: a mood. Verbatim
 * text and spatial layout are what a text-only model can act on.
 */
const DESCRIBE_PROMPT = [
	"Read the image at the path given and write a description for an assistant that cannot see images.",
	"Include, in this order:",
	"1. Every piece of visible text, transcribed exactly, keeping line breaks, in a fenced block.",
	"2. What kind of screen this is and what state it is in.",
	"3. Layout and spatial relationships: what is where, what is selected, what is highlighted.",
	"4. Colours, and any red/error/green/success colouring that carries meaning.",
	"Be literal. Do not speculate about intent and do not advise what to do about it.",
].join("\n");

interface TextBlock {
	type: "text";
	text: string;
}
interface ImageBlock {
	type: "image";
	data: string;
	mimeType: string;
}
type Block = TextBlock | ImageBlock;

/** The read tool's own note, which is wrong once we supply a description. */
const NON_VISION_NOTE = "does not support images";

function modelSeesImages(model: { id?: string; input?: unknown } | undefined): boolean {
	if (FORCE) return false;
	if (!model) return true;
	const input = model.input;
	return Array.isArray(input) && (input as string[]).includes("image");
}

function hashOf(text: string): string {
	return createHash("sha256").update(text).digest("hex").slice(0, 16);
}

function extFor(mimeType: string): string {
	const map: Record<string, string> = {
		"image/png": ".png",
		"image/jpeg": ".jpg",
		"image/gif": ".gif",
		"image/webp": ".webp",
	};
	return map[mimeType] ?? ".png";
}

/**
 * Put the bytes on disk so a child `pi` can hand them to a vision model through
 * the read tool. Pasted images arrive as base64 with no file behind them, so
 * this is also what gives `inspect_image` something to point at.
 */
function materialise(data: string, mimeType: string, key: string): string {
	mkdirSync(CACHE_DIR, { recursive: true });
	const file = join(CACHE_DIR, `${key}${extFor(mimeType)}`);
	if (!existsSync(file)) writeFileSync(file, Buffer.from(data, "base64"));
	return file;
}

function cachedDescription(key: string): string | undefined {
	const file = join(CACHE_DIR, `${key}.txt`);
	if (!existsSync(file)) return undefined;
	try {
		return readFileSync(file, "utf8").trim() || undefined;
	} catch {
		return undefined;
	}
}

function cacheDescription(key: string, text: string): void {
	try {
		mkdirSync(CACHE_DIR, { recursive: true });
		writeFileSync(join(CACHE_DIR, `${key}.txt`), text);
	} catch {
		// a cache miss next time is not worth failing a request over
	}
}

/**
 * Extensions the child needs that `--no-extensions` would otherwise hide.
 *
 * `--no-extensions` is what stops this extension loading itself and recursing,
 * but it also hides anything that registers a provider or patches one. Both
 * matter here: the opencode-accountN providers come from multi-account, and
 * without the opencode-free-tier header patch every free-tier model answers
 * 403. The child also needs the full builtin tool list for the same reason --
 * opencode's free tier refuses a request whose tool list is not the official
 * client one, so a read-only child is rejected.
 */
function childExtensionArgs(): string[] {
	const configured = (process.env.PI_IMAGE_COMPACT_CHILD_EXTENSIONS ?? "")
		.split(",")
		.map((s) => s.trim())
		.filter(Boolean);
	const wanted =
		configured.length > 0
			? configured
			: [join(AGENT_DIR, "extensions", "multi-account"), join(AGENT_DIR, "extensions", "opencode-free-tier")];
	const args: string[] = [];
	for (const entry of wanted) {
		const resolved = resolveEntry(entry);
		if (resolved) args.push("--extension", resolved);
	}
	return args;
}

/** read,write,edit,bash,grep,find,ls — the set the free tier accepts. */
const CHILD_TOOLS =
	process.env.PI_IMAGE_COMPACT_CHILD_TOOLS?.trim() || "read,write,edit,bash,grep,find,ls";

/** An extension entry may be the file itself or a directory holding an index. */
function resolveEntry(entry: string): string | undefined {
	for (const candidate of ["index.ts", "index.js", "index.mjs"]) {
		const file = join(entry, candidate);
		if (existsSync(file)) return file;
	}
	return existsSync(entry) ? entry : undefined;
}

/** Run one nested pi turn that reads the image and describes it. */
function describeWithVisionModel(imagePath: string, prompt: string): Promise<string> {
	const slash = VISION_MODEL.indexOf("/");
	const provider = slash > 0 ? VISION_MODEL.slice(0, slash) : undefined;
	const model = slash > 0 ? VISION_MODEL.slice(slash + 1) : VISION_MODEL;

	return new Promise((resolve) => {
		const args = [
			"-p",
			"--no-session",
			// keeps this extension out of its own child, so it cannot recurse
			"--no-extensions",
			"--no-prompt-templates",
			"--no-themes",
			"--tools",
			CHILD_TOOLS,
			...childExtensionArgs(),
		];
		if (provider) args.push("--provider", provider);
		args.push("--model", model);
		args.push(`${prompt}\n\nImage path: ${imagePath}`);

		const child = spawn("pi", args, { stdio: ["ignore", "pipe", "pipe"] });
		let out = "";
		let err = "";
		const timer = setTimeout(() => child.kill("SIGKILL"), TIMEOUT_MS);

		child.stdout.on("data", (d) => {
			out += String(d);
		});
		child.stderr.on("data", (d) => {
			err += String(d);
		});
		child.on("error", (e) => {
			clearTimeout(timer);
			resolve(`[image description unavailable: could not run the vision model (${String(e)})]`);
		});
		child.on("close", (code) => {
			clearTimeout(timer);
			const text = out.trim();
			if (text) resolve(text);
			else if (code === 0) resolve("[image description unavailable: the vision model returned nothing]");
			else resolve(`[image description unavailable: the vision model failed (exit ${code}). ${err.trim().slice(0, 200)}]`);
		});
	});
}

/** Describe one image, cached on its bytes and the prompt that asked for it. */
async function describeImage(block: ImageBlock, prompt: string): Promise<{ text: string; file: string }> {
	const key = hashOf(`${prompt} ${block.data}`);
	const hit = cachedDescription(key);
	if (hit) {
		// the file may have been cleared while the description survived
		return { text: hit, file: materialise(block.data, block.mimeType, key) };
	}
	let file: string;
	try {
		file = materialise(block.data, block.mimeType, key);
	} catch (e) {
		return { text: `[image description unavailable: could not write the image to disk (${String(e)})]`, file: "" };
	}
	const text = await describeWithVisionModel(file, prompt);
	if (!text.startsWith("[image description unavailable")) cacheDescription(key, text);
	return { text, file };
}

function renderBlock(index: number, count: number, file: string, description: string): string {
	const which = count > 1 ? ` ${index + 1} of ${count}` : "";
	return [
		`[image${which} — ${file}]`,
		`The active model cannot accept images, so a vision model described this one instead.`,
		`If the description is not specific enough, call inspect_image with the path above and a focus.`,
		`Do not assume it is complete.`,
		"",
		description.trim(),
	].join("\n");
}

/**
 * Replace every image in a content array with a description.
 * Returns undefined when there was nothing to do, so callers can leave the
 * original content untouched.
 */
async function convertBlocks(
	blocks: Block[],
	opts: { dropNonVisionNote?: boolean } = {},
): Promise<Block[] | undefined> {
	const images = blocks.filter((b): b is ImageBlock => b?.type === "image");
	if (images.length === 0) return undefined;

	// Describe in parallel: two screenshots should not cost the sum of two calls.
	const described = await Promise.all(images.map((b) => describeImage(b, DESCRIBE_PROMPT)));

	const replacements = new Map<Block, string>();
	described.forEach((d, i) => replacements.set(images[i], renderBlock(i, images.length, d.file, d.text)));

	const out: Block[] = [];
	let converted = false;
	for (const block of blocks) {
		const text = replacements.get(block as ImageBlock);
		if (text !== undefined) {
			out.push({ type: "text", text });
			converted = true;
			continue;
		}
		// pi's read tool adds a note saying the image will be omitted. Once we
		// have supplied a description that note is simply wrong, and leaving it
		// in tells the model to distrust the text it is reading.
		if (
			converted &&
			opts.dropNonVisionNote &&
			block?.type === "text" &&
			(block as TextBlock).text.includes(NON_VISION_NOTE)
		) {
			continue;
		}
		out.push(block);
	}
	return converted ? out : undefined;
}

export default function imageCompact(pi: ExtensionAPI): void {
	/**
	 * Images produced by a tool — the read tool on a png, jpg, gif, webp or bmp.
	 * Replacing the content here means the picture never reaches history, so
	 * `downgradeUnsupportedImages` never has a chance to throw it away.
	 */
	pi.on("tool_result", async (event, ctx) => {
		if (modelSeesImages(ctx.model)) return;
		const content = event.content as unknown as Block[];
		if (!Array.isArray(content)) return;
		const converted = await convertBlocks(content, { dropNonVisionNote: true });
		if (!converted) return;
		return { content: converted as never, details: (event as { details?: unknown }).details };
	});

	/**
	 * Images the user pasted or attached, which arrive as a user message rather
	 * than a tool result. `context` runs before every LLM call with the
	 * transcript intact, so the bytes are still here. Descriptions are cached,
	 * so re-running this each turn is a cache read, not a model call.
	 */
	pi.on("context", async (event, ctx) => {
		if (modelSeesImages(ctx.model)) return;
		const messages = event.messages as unknown as { role?: string; content?: unknown }[];
		if (!Array.isArray(messages)) return;

		let touched = false;
		const next = [];
		for (const message of messages) {
			const content = message?.content;
			if (!Array.isArray(content) || !content.some((b) => (b as Block)?.type === "image")) {
				next.push(message);
				continue;
			}
			const converted = await convertBlocks(content as Block[]);
			if (!converted) {
				next.push(message);
				continue;
			}
			touched = true;
			next.push({ ...message, content: converted });
		}
		return touched ? { messages: next as never } : undefined;
	});

	/**
	 * The fallback. Resending the image would not help a model that cannot
	 * decode it, so this sends a vision model back to the same file with a
	 * focus and returns a second, targeted description.
	 */
	pi.registerTool({
		name: "inspect_image",
		label: "Inspect image",
		description:
			"Describe an image again, in more detail, using a vision model. Use this when an image's " +
			"description was not specific enough — pass a focus such as \"the error dialog in the top " +
			"right\" or \"the value in the second column\". Works even though the active model cannot " +
			"see images itself.",
		parameters: Type.Object({
			path: Type.String({ description: "Image path, as given in the description marker." }),
			focus: Type.Optional(Type.String({ description: "What to look at, if only part of it matters." })),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
			const file = params.path;
			if (!existsSync(file)) {
				return { content: [{ type: "text" as const, text: `No such file: ${file}` }], isError: true, details: {} };
			}
			const focus = params.focus?.trim();
			const prompt = focus
				? `${DESCRIBE_PROMPT}\n\nThis time, focus on: ${focus}. Be exhaustive about it.`
				: DESCRIBE_PROMPT;
			let data: string;
			try {
				data = readFileSync(file).toString("base64");
			} catch (e) {
				return {
					content: [{ type: "text" as const, text: `Could not read ${file}: ${String(e)}` }],
					isError: true,
					details: {},
				};
			}
			const ext = extname(file).toLowerCase();
			const mime = ext === ".jpg" || ext === ".jpeg" ? "image/jpeg" : `image/${ext.slice(1) || "png"}`;
			const { text } = await describeImage({ type: "image", data, mimeType: mime }, prompt);
			return { content: [{ type: "text" as const, text }], details: {} };
		},
	});
}
