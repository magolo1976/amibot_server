import { env as hf, StyleTextToSpeech2Model, AutoTokenizer, Tensor, RawAudio } from "@huggingface/transformers";
import { phonemize } from "./phonemize.js";
import { TextSplitterStream } from "./splitter.js";
import { getVoiceData, VOICES } from "./voices.js";

const STYLE_DIM = 256;
const SAMPLE_RATE = 24000;

export function cleanText(input) {
  if (typeof input !== "string") return "";
  return input
    .replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, "")
    .replace(/[*#`_~^|\\<>[\]{}@¡¿]/g, "")
    .replace(/[‘’]/g, "'")
    .replace(/[“”«»]/g, '"')
    .replace(/[–—]/g, "-")
    .replace(/…/g, ".")
    .replace(/\s+/g, " ")
    .trim();
}

export class TTSComms {
  constructor(model, tokenizer) {
    this.model = model;
    this.tokenizer = tokenizer;
  }

  static async from_pretrained(model_id, { dtype = "fp32", device = null, progress_callback = null } = {}) {
    const model = StyleTextToSpeech2Model.from_pretrained(model_id, { progress_callback, dtype, device });
    const tokenizer = AutoTokenizer.from_pretrained(model_id, { progress_callback });

    const info = await Promise.all([model, tokenizer]);
    return new TTSComms(...info);
  }

  get voices() {
    return VOICES;
  }

  list_voices() {
    console.table(VOICES);
  }

  _validate_voice(voice) {
    if (!VOICES.hasOwnProperty(voice)) {
      throw new Error(`Voice "${voice}" not found. Should be one of: ${Object.keys(VOICES).join(", ")}.`);
    }
    return voice.at(0);
  }

  async generate(text, { voice = "ef_dora", speed = 1 } = {}) {
    const language = this._validate_voice(voice);
    const cleaned = cleanText(text);
    const phonemes = await phonemize(cleaned, language);
    const { input_ids } = this.tokenizer(phonemes, {
      truncation: true,
    });

    return this.generate_from_ids(input_ids, { voice, speed });
  }

  async generate_from_ids(input_ids, { voice = "ef_dora", speed = 1 } = {}) {
    const num_tokens = Math.min(Math.max(input_ids.dims.at(-1) - 2, 0), 509);

    const data = await getVoiceData(voice);
    const offset = num_tokens * STYLE_DIM;
    const voiceData = data.slice(offset, offset + STYLE_DIM);

    const inputs = {
      input_ids,
      style: new Tensor("float32", voiceData, [1, STYLE_DIM]),
      speed: new Tensor("float32", [speed], [1]),
    };

    const { waveform } = await this.model(inputs);
    return new RawAudio(waveform.data, SAMPLE_RATE);
  }

  async *stream(text, { voice = "ef_dora", speed = 1, split_pattern = null } = {}) {
    const language = this._validate_voice(voice);

    let splitter;
    if (text instanceof TextSplitterStream) {
      splitter = text;
    } else if (typeof text === "string") {
      const cleaned = cleanText(text);
      splitter = new TextSplitterStream();
      const chunks = split_pattern
        ? cleaned
            .split(split_pattern)
            .map((chunk) => chunk.trim())
            .filter((chunk) => chunk.length > 0)
        : [cleaned];
      splitter.push(...chunks);
      splitter.close();
    } else {
      throw new Error("Invalid input type. Expected string or TextSplitterStream.");
    }
    for await (const sentence of splitter) {
      const phonemes = await phonemize(sentence, language);
      const { input_ids } = this.tokenizer(phonemes, {
        truncation: true,
      });

      const audio = await this.generate_from_ids(input_ids, { voice, speed });
      yield { text: sentence, phonemes, audio };
    }
  }
}

export const env = {
  set cacheDir(value) {
    hf.cacheDir = value;
  },
  get cacheDir() {
    return hf.cacheDir;
  },
  set wasmPaths(value) {
    hf.backends.onnx.wasm.wasmPaths = value;
  },
  get wasmPaths() {
    return hf.backends.onnx.wasm.wasmPaths;
  },
};

export { TextSplitterStream, VOICES, getVoiceData };
