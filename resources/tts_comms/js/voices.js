import path from "path";
import fs from "fs/promises";

export const VOICES = Object.freeze({
  af_heart: { name: "Heart", language: "en-us", gender: "Female", targetQuality: "A", overallGrade: "A" },
  af_alloy: { name: "Alloy", language: "en-us", gender: "Female", targetQuality: "B", overallGrade: "C" },
  af_aoede: { name: "Aoede", language: "en-us", gender: "Female", targetQuality: "B", overallGrade: "C+" },
  af_bella: { name: "Bella", language: "en-us", gender: "Female", targetQuality: "A", overallGrade: "A-" },
  af_jessica: { name: "Jessica", language: "en-us", gender: "Female", targetQuality: "C", overallGrade: "D" },
  af_kore: { name: "Kore", language: "en-us", gender: "Female", targetQuality: "B", overallGrade: "C+" },
  af_nicole: { name: "Nicole", language: "en-us", gender: "Female", targetQuality: "B", overallGrade: "B-" },
  af_nova: { name: "Nova", language: "en-us", gender: "Female", targetQuality: "B", overallGrade: "C" },
  af_river: { name: "River", language: "en-us", gender: "Female", targetQuality: "C", overallGrade: "D" },
  af_sarah: { name: "Sarah", language: "en-us", gender: "Female", targetQuality: "B", overallGrade: "C+" },
  af_sky: { name: "Sky", language: "en-us", gender: "Female", targetQuality: "B", overallGrade: "C-" },
  am_adam: { name: "Adam", language: "en-us", gender: "Male", targetQuality: "D", overallGrade: "F+" },
  am_echo: { name: "Echo", language: "en-us", gender: "Male", targetQuality: "C", overallGrade: "D" },
  am_eric: { name: "Eric", language: "en-us", gender: "Male", targetQuality: "C", overallGrade: "D" },
  am_fenrir: { name: "Fenrir", language: "en-us", gender: "Male", targetQuality: "B", overallGrade: "C+" },
  am_liam: { name: "Liam", language: "en-us", gender: "Male", targetQuality: "C", overallGrade: "D" },
  am_michael: { name: "Michael", language: "en-us", gender: "Male", targetQuality: "B", overallGrade: "C+" },
  am_onyx: { name: "Onyx", language: "en-us", gender: "Male", targetQuality: "C", overallGrade: "D" },
  am_puck: { name: "Puck", language: "en-us", gender: "Male", targetQuality: "B", overallGrade: "C+" },
  am_santa: { name: "Santa", language: "en-us", gender: "Male", targetQuality: "C", overallGrade: "D-" },
  bf_alice: { name: "Alice", language: "en-gb", gender: "Female", targetQuality: "C", overallGrade: "D" },
  bf_emma: { name: "Emma", language: "en-gb", gender: "Female", targetQuality: "B", overallGrade: "B-" },
  bf_isabella: { name: "Isabella", language: "en-gb", gender: "Female", targetQuality: "B", overallGrade: "C" },
  bf_lily: { name: "Lily", language: "en-gb", gender: "Female", targetQuality: "C", overallGrade: "D" },
  bm_daniel: { name: "Daniel", language: "en-gb", gender: "Male", targetQuality: "C", overallGrade: "D" },
  bm_fable: { name: "Fable", language: "en-gb", gender: "Male", targetQuality: "B", overallGrade: "C" },
  bm_george: { name: "George", language: "en-gb", gender: "Male", targetQuality: "B", overallGrade: "C" },
  bm_lewis: { name: "Lewis", language: "en-gb", gender: "Male", targetQuality: "C", overallGrade: "D+" },
  ef_dora: { name: "Dora", language: "es", gender: "Female", targetQuality: "B", overallGrade: "B" },
  em_alex: { name: "Alex", language: "es", gender: "Male", targetQuality: "B", overallGrade: "B" },
  em_santa: { name: "Santa ES", language: "es", gender: "Male", targetQuality: "C", overallGrade: "C" },
  ff_siwis: { name: "Siwis", language: "fr-fr", gender: "Female", targetQuality: "B", overallGrade: "B-" },
  hf_alpha: { name: "Alpha", language: "hi", gender: "Female", targetQuality: "B", overallGrade: "C" },
  hf_beta: { name: "Beta", language: "hi", gender: "Female", targetQuality: "B", overallGrade: "C" },
  hm_omega: { name: "Omega", language: "hi", gender: "Male", targetQuality: "B", overallGrade: "C" },
  hm_psi: { name: "Psi", language: "hi", gender: "Male", targetQuality: "B", overallGrade: "C" },
  if_sara: { name: "Sara", language: "it", gender: "Female", targetQuality: "B", overallGrade: "C" },
  im_nicola: { name: "Nicola", language: "it", gender: "Male", targetQuality: "B", overallGrade: "C" },
  jf_alpha: { name: "Alpha JA", language: "ja", gender: "Female", targetQuality: "B", overallGrade: "C+" },
  jf_gongitsune: { name: "Gongitsune", language: "ja", gender: "Female", targetQuality: "B", overallGrade: "C" },
  jf_nezumi: { name: "Nezumi", language: "ja", gender: "Female", targetQuality: "B", overallGrade: "C-" },
  jf_tebukuro: { name: "Tebukuro", language: "ja", gender: "Female", targetQuality: "B", overallGrade: "C" },
  jm_kumo: { name: "Kumo", language: "ja", gender: "Male", targetQuality: "B", overallGrade: "C-" },
  pf_dora: { name: "Dora PT", language: "pt-br", gender: "Female", targetQuality: "C", overallGrade: "D" },
  pm_alex: { name: "Alex PT", language: "pt-br", gender: "Male", targetQuality: "C", overallGrade: "D" },
  pm_santa: { name: "Santa PT", language: "pt-br", gender: "Male", targetQuality: "C", overallGrade: "D" },
  zf_xiaobei: { name: "Xiaobei", language: "zh", gender: "Female", targetQuality: "C", overallGrade: "D" },
  zf_xiaoni: { name: "Xiaoni", language: "zh", gender: "Female", targetQuality: "C", overallGrade: "D" },
  zf_xiaoxiao: { name: "Xiaoxiao", language: "zh", gender: "Female", targetQuality: "C", overallGrade: "D" },
  zf_xiaoyi: { name: "Xiaoyi", language: "zh", gender: "Female", targetQuality: "C", overallGrade: "D" },
  zm_yunjian: { name: "Yunjian", language: "zh", gender: "Male", targetQuality: "C", overallGrade: "D" },
  zm_yunxi: { name: "Yunxi", language: "zh", gender: "Male", targetQuality: "C", overallGrade: "D" },
  zm_yunxia: { name: "Yunxia", language: "zh", gender: "Male", targetQuality: "C", overallGrade: "D" },
  zm_yunyang: { name: "Yunyang", language: "zh", gender: "Male", targetQuality: "C", overallGrade: "D" }
});

async function getVoiceFile(id) {
  const dirname = typeof __dirname !== "undefined" ? __dirname : import.meta.dirname;
  const candidates = [
    path.resolve(dirname, `voices/${id}.bin`),
    path.resolve(dirname, `../voices/${id}.bin`),
    `/app/voices/${id}.bin`,
    `C:/_PROYECTOS_2030/amibot/amibot_server/libraries/tts_comms/voices/${id}.bin`,
    `C:/_PROYECTOS_2030/amibot/amibot_server/resources/tts_comms/voices/${id}.bin`,
  ];
  for (const file of candidates) {
    try {
      const { buffer } = await fs.readFile(file);
      return buffer;
    } catch (_) {
      // try next path
    }
  }
  throw new Error(`Voice file not found for id: ${id}`);
}

const VOICE_CACHE = new Map();
export async function getVoiceData(voice) {
  if (VOICE_CACHE.has(voice)) {
    return VOICE_CACHE.get(voice);
  }

  const buffer = new Float32Array(await getVoiceFile(voice));
  VOICE_CACHE.set(voice, buffer);
  return buffer;
}
