import React, { useEffect, useRef, useState } from 'react';
import { VideoMetaMP } from '../data/videometa';

interface VideoCardProps {
  videometa: VideoMetaMP;
  liquidGlassCard: React.CSSProperties;
  liquidGlassCardHover: React.CSSProperties;
}

const MPVideoCard: React.FC<VideoCardProps> = ({ videometa, liquidGlassCard, liquidGlassCardHover }) => {

  const videoRef = useRef<HTMLVideoElement>(null);
  const captionRefs = useRef<(HTMLLIElement | null)[]>([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [activeIndex, setActiveIndex] = useState(0);

  // 处理视频时间更新
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    
    const handleTimeUpdate = () => {
      const currentVideoTime = video.currentTime;
      setCurrentTime(currentVideoTime);
      
      // 找到当前应该高亮的字幕索引
      const prompts = videometa.prompts;
      let newIndex = 0;
      
      // 从后向前查找，找到第一个时间小于当前时间的字幕
      for (let i = prompts.length - 1; i >= 0; i--) {
        if (currentVideoTime >= prompts[i].time) {
          newIndex = i;
          break;
        }
      }
      
      setActiveIndex(newIndex);
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    
    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, []);

  // 当 activeIndex 变化时，滚动到对应字幕
  // useEffect(() => {
  //   if (activeIndex >= 0 && captionRefs.current[activeIndex]) {
  //     const activeElement = captionRefs.current[activeIndex];
      
  //     // 使用平滑滚动效果
  //     activeElement?.scrollIntoView({
  //       behavior: 'smooth',
  //       block: 'nearest',
  //       inline: 'nearest'
  //     });
  //   }
  // }, [activeIndex]);

  // 点击字幕跳转到对应时间点
  const handleCaptionClick = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      videoRef.current.play();
    }
  };

  // 格式化时间（秒 → 分:秒）
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  return (
    <div
      // key={videometa.id}
      style={{
        ...liquidGlassCard,
        overflow: 'hidden',
        cursor: 'pointer',
        padding: '0'
      }}
      onMouseEnter={(e) => {
        Object.assign(e.currentTarget.style, liquidGlassCardHover);
      }}
      onMouseLeave={(e) => {
        Object.assign(e.currentTarget.style, liquidGlassCard);
      }}
    >
      {/* 视频 */}
      <div style={{ position: 'relative' }}>
        <video
          ref={videoRef}
          src={videometa.url}
          style={{ width: '100%', height: '380px', objectFit: 'cover' }}
          muted
          controls
          // loop
        />
        {/* <div style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          background: 'linear-gradient(135deg, #FFD000 0%,rgb(255, 160, 51) 100%)',
          color: '#000000',
          padding: '4px 8px',
          borderRadius: '12px',
          fontSize: '10px',
          fontWeight: '700',
          textTransform: 'uppercase',
          letterSpacing: '1px',
          boxShadow: '0 4px 15px rgba(255, 208, 0, 0.4)'
        }}>
          {videometa.label}
        </div> */}

        {/* 当前时间显示 */}
        <div className="absolute bottom-3 right-3 bg-black bg-opacity-70 text-white px-2 py-1 rounded text-xs">
          {formatTime(currentTime)}
        </div>
      </div>

      {/* 文字 */}
      <div className="p-4">
        {/* 视频标题 */}
        {/* <h3 className="text-xl font-semibold mb-2">{category.title}</h3> */}
        
        {/* 字幕列表 */}
        {/* <div className="mt-4 bg-gray-50 rounded-lg p-3 max-h-60 overflow-y-auto"> */}
        <div className="my-4 p-4 max-h-80 overflow-y-auto" style={{
            // backgroundColor: '#374151',
            // borderRadius: '8px',
            // overflow: 'hidden',
            // transition: 'transform 0.3s',
            cursor: 'pointer'
        }}>
          {/* <h4 className="font-medium text-gray-700 mb-2">视频字幕</h4> */}
          <ul className="space-y-2">
            {videometa.prompts.map((prompt, index) => (
              <li 
                key={index}
                ref={el => captionRefs.current[index] = el}
                onClick={() => handleCaptionClick(prompt.time)}
                className={`
                  p-1 cursor-pointer transition-all duration-300 ease-in-out rounded
                  transform origin-left overflow-hidden
                  ${index === activeIndex 
                    ? 'bg-gradient-to-r from-amber-400 to-orange-500 scale-[1.02]' 
                    : 'hover:bg-gray-100'}
                `}
              >
                <div className="flex items-start gap-2">
                  <span className={`
                    text-xs px-2 py-0.5 mt-0.5 ml-1 rounded transition-colors duration-300 bg-gray-200 text-gray-700
                  `}>
                    {formatTime(prompt.time)}
                  </span>
                  <span 
                    className={`
                      transition-colors duration-300 flex-1
                      ${index === activeIndex ? 'text-white' : 'text-gray-700'}
                      break-words
                    `} 
                    style={{ fontSize: '14px', padding: '1px' }}
                  >
                    {prompt.text}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default MPVideoCard;