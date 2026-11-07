import { TTSComms, env, cleanText } from "./tts_comms.js";
import { split } from "./splitter.js";
import { RawAudio } from "@huggingface/transformers";
import path from "path";
import fs from "fs";
import readline from "readline";

const args = process.argv.slice(2);
const isWorker = args[0] === "--worker";

const candidates = [
  isWorker ? args[1] : args[4],
  path.resolve(import.meta.dirname, "../../../model/tts_onnx"),
  path.resolve(import.meta.dirname, "../../model/tts_onnx"),
  "C:/_PROYECTOS_2030/amibot/amibot_server/model/tts_onnx",
  "/app/models/tts_onnx"
].filter(Boolean);

for (const c of candidates) {
  if (fs.existsSync(c)) {
    env.cacheDir = c;
    break;
  }
}

const model_id = "onnx-community/Kokoro-82M-v1.0-ONNX";
const tts = await TTSComms.from_pretrained(model_id, { dtype: "q8", device: "cpu" });

async function synthesizeFloat32(rawText, voice, speed) {
  const cleaned = cleanText(rawText || "");
  const sentences = split(cleaned).filter((s) => s.trim().length > 0);
  if (sentences.length === 0 && cleaned.length > 0) {
    sentences.push(cleaned);
  }
  const chunks = [];
  for (const sentence of sentences) {
    const audio = await tts.generate(sentence, { voice, speed });
    if (audio && audio.audio && audio.audio.length > 0) {
      chunks.push(audio.audio);
    }
  }
  const totalLen = chunks.reduce((acc, arr) => acc + arr.length, 0);
  const merged = new Float32Array(totalLen);
  let offset = 0;
  for (const arr of chunks) {
    merged.set(arr, offset);
    offset += arr.length;
  }
  return merged;
}

if (isWorker) {
  // Modo Worker persistente en RAM: comunica por pipes stdin/stdout sin tocar el disco jamas
  const writeBufferAll = (buf) =>
    new Promise((resolve, reject) => {
      process.stdout.write(buf, (err) => (err ? reject(err) : resolve()));
    });

  await writeBufferAll(Buffer.from("RDY\n", "ascii"));

  const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity
  });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const req = JSON.parse(trimmed);
      if (req.cmd === "quit") break;
      const merged = await synthesizeFloat32(
        req.text || "",
        req.voice || "ef_dora",
        typeof req.speed === "number" && req.speed > 0 ? req.speed : 1.0
      );
      const hdr = Buffer.alloc(8);
      hdr.write("PCM1", 0, 4, "ascii");
      hdr.writeUInt32LE(merged.length, 4);
      const payload = Buffer.from(merged.buffer, merged.byteOffset, merged.byteLength);
      await writeBufferAll(hdr);
      if (payload.length > 0) {
        await writeBufferAll(payload);
      }
    } catch {
      const hdr = Buffer.alloc(8);
      hdr.write("PCM1", 0, 4, "ascii");
      hdr.writeUInt32LE(0, 4);
      await writeBufferAll(hdr);
    }
  }
  process.exit(0);
} else {
  let textArg = args[0] || "";
  const outWav = args[1] || "out.wav";
  const voice = args[2] || "ef_dora";
  const speed = parseFloat(args[3] || "1.0");
  const merged = await synthesizeFloat32(textArg, voice, speed);
  const finalAudio = new RawAudio(merged, 24000);
  await finalAudio.save(outWav);
}
