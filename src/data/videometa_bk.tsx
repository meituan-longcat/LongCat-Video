export interface VideoMeta {
    id: string;
    label: string;
    title: string;
    videoId: string;
    prompt: string;
  }
  
const videometas: VideoMeta[] = [
    { id: 'action', label: 'Action', title: 'Dynamic Action Sequences', videoId: '2397192940', prompt: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.' },
    { id: 'animation', label: 'Animation', title: 'Fluid Animation', videoId: '1989921539', prompt: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
    { id: 'fantasy', label: 'Fantasy', title: 'Fantasy Worlds', videoId: '3129405062', prompt: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.' },
    { id: 'landscape', label: 'Landscape', title: 'Natural Landscapes', videoId: '2526908098', prompt: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.' },
    { id: 'portrait', label: 'Portrait', title: 'Character Portraits', videoId: '3542015254', prompt: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.' },
    { id: 'cinematic', label: 'Cinematic', title: 'Professional Cinematic Quality', videoId: '3641332521', prompt: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.' }
];
  
const videometas_hs: VideoMeta[] = [
    { id: 'action', label: 'Action', title: 'Dynamic Action Sequences', videoId: '2397192940', prompt: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.' },
    { id: 'animation', label: 'Animation', title: 'Fluid Animation', videoId: '1989921539', prompt: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
    { id: 'fantasy', label: 'Fantasy', title: 'Fantasy Worlds', videoId: '3129405062', prompt: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.' },
    { id: 'landscape', label: 'Landscape', title: 'Natural Landscapes', videoId: '2526908098', prompt: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.' },
    { id: 'portrait', label: 'Portrait', title: 'Character Portraits', videoId: '3542015254', prompt: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.' },
    { id: 'cinematic', label: 'Cinematic', title: 'Professional Cinematic Quality', videoId: '3641332521', prompt: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.' }
];

interface TimedPrompt {
    time: number; // 时间戳（秒）
    text: string; // 字幕文本
}
  

export interface VideoMetaMP {
  id: string;
  videoId: string;
  label: string;
  title: string;
  prompts: TimedPrompt[];
}

const videometas_mp: VideoMetaMP[] = [
    { 
        id: 'action', 
        label: 'Action', 
        title: 'Dynamic Action Sequences', 
        videoId: '2397192940', 
        prompts: [
           { time: 0, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
           { time: 3, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
           { time: 6, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
        ] 
    },
    { 
        id: 'animation', 
        label: 'Animation', 
        title: 'Fluid Animation', 
        videoId: '1989921539', 
        prompts: [
            { time: 0, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
            { time: 3, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
            { time: 6, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
        ] 
    },
    { 
        id: 'fantasy', 
        label: 'Fantasy', 
        title: 'Fantasy Worlds', 
        videoId: '3129405062', 
        prompts: [
            { time: 0, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
            { time: 3, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
            { time: 6, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
        ] 
    },
    { 
        id: 'landscape', 
        label: 'Landscape', 
        title: 'Natural Landscapes', 
        videoId: '2526908098', 
        prompts: [
            { time: 0, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
            { time: 3, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
            { time: 6, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
        ]  
    },
    { 
        id: 'portrait', 
        label: 'Portrait', 
        title: 'Character Portraits', 
        videoId: '3542015254', 
        prompts: [
            { time: 0, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
            { time: 3, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
            { time: 6, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
        ] 
    },
    { 
        id: 'cinematic', 
        label: 'Cinematic', 
        title: 'Professional Cinematic Quality', 
        videoId: '3641332521', 
        prompts: [
            { time: 0, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
            { time: 3, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
            { time: 6, text: 'realistic photography, A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage.' },
        ] 
    }
];
  
  // 导出所有内容
export { videometas, videometas_hs, videometas_mp };
//   export default videometas; // 默认导出第一个数组