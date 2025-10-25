import React, { useState, useRef, useEffect } from 'react';
import VideoCard from '../components/video_card';
import { VideoMeta } from '../data/videometa';

interface VideoGallerySectionProps {
  sectionId: string;
  sectionStyle: React.CSSProperties;
  containerStyle: React.CSSProperties;
  liquidGlassCard: React.CSSProperties;
  liquidGlassCardHover: React.CSSProperties;
  galleryGroups: Array<{
    title: string;
    videometas: VideoMeta[];
  }>;
}

const HSRecurVideoGallerySection2: React.FC<VideoGallerySectionProps> = ({
  sectionId,
  sectionStyle,
  containerStyle,
  liquidGlassCard,
  liquidGlassCardHover,
  galleryGroups
}) => {
  const [currentGroupIndex, setCurrentGroupIndex] = useState(0);
  const galleryRef = useRef<HTMLDivElement>(null);
  
  // 处理滚动事件
  const scrollGallery = (direction: 'left' | 'right') => {
    if (direction === 'left') {
      setCurrentGroupIndex(prev => Math.max(0, prev - 1));
    } else {
      setCurrentGroupIndex(prev => Math.min(galleryGroups.length - 1, prev + 1));
    }
  };

  // 当索引变化时滚动到对应位置
  useEffect(() => {
    if (galleryRef.current) {
      const scrollContainer = galleryRef.current;
      const scrollWidth = scrollContainer.scrollWidth;
      const groupWidth = scrollWidth / galleryGroups.length;
      
      scrollContainer.scrollTo({
        left: groupWidth * currentGroupIndex,
        behavior: 'smooth'
      });
    }
  }, [currentGroupIndex, galleryGroups.length]);

  // 按钮样式
  const buttonStyle: React.CSSProperties = {
    position: 'absolute',
    top: '50%',
    transform: 'translateY(-50%)',
    width: '50px',
    height: '50px',
    borderRadius: '50%',
    background: 'rgba(255, 255, 255, 0.9)',
    border: 'none',
    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 10,
    transition: 'all 0.3s ease',
  };

  return (
    <section id={sectionId} style={{ ...sectionStyle, position: 'relative', zIndex: 5 }}>
      <div style={containerStyle}>
        <div style={{ textAlign: 'center', marginBottom: '60px' }}>
          <h2 style={{ 
            fontSize: '3rem', 
            fontWeight: 'bold', 
            marginBottom: '20px', 
            color: '#1d1d1f' 
          }}>
            Image-to-Video Gallery
          </h2>
          <p style={{ 
            color: '#86868b', 
            maxWidth: '800px', 
            margin: '0 auto', 
            fontSize: '18px', 
            lineHeight: '1.6' 
          }}>
            Explore diverse video demonstrations showcasing LongCat-Video's versatility across
            different genres, from cinematic storytelling to realistic physics simulations.
          </p>
        </div>

        {/* 画廊容器 */}
        <div style={{ position: 'relative', marginBottom: '40px' }}>
          {/* 左箭头 */}
          <button
            onClick={() => scrollGallery('left')}
            style={{
              ...buttonStyle,
              left: '-60px',
              opacity: currentGroupIndex === 0 ? 0.5 : 1,
              pointerEvents: currentGroupIndex === 0 ? 'none' : 'auto'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-50%) scale(1.1)';
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.95)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(-50%) scale(1)';
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.9)';
            }}
          >
            <div style={{
              width: 0,
              height: 0,
              borderRight: '12px solid #1d1d1f',
              borderTop: '8px solid transparent',
              borderBottom: '8px solid transparent',
              marginRight: '2px'
            }} />
          </button>

          {/* 右箭头 */}
          <button
            onClick={() => scrollGallery('right')}
            style={{
              ...buttonStyle,
              right: '-60px',
              opacity: currentGroupIndex >= galleryGroups.length - 1 ? 0.5 : 1,
              pointerEvents: currentGroupIndex >= galleryGroups.length - 1 ? 'none' : 'auto'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-50%) scale(1.1)';
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.95)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(-50%) scale(1)';
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.9)';
            }}
          >
            <div style={{
              width: 0,
              height: 0,
              borderLeft: '12px solid #1d1d1f',
              borderTop: '8px solid transparent',
              borderBottom: '8px solid transparent',
              marginLeft: '2px'
            }} />
          </button>

          {/* 横向滚动画廊 */}
          <div 
            ref={galleryRef}
            style={{
              display: 'flex',
              overflowX: 'hidden',
              scrollBehavior: 'smooth',
              borderRadius: '16px',
              position: 'relative',
              padding: '10px 0'
            }}
          >
            {galleryGroups.map((group, index) => (
              <div 
                key={index}
                style={{
                  minWidth: '100%',
                  padding: '0 20px',
                  boxSizing: 'border-box'
                }}
              >
                {/* <h3 style={{
                  fontSize: '1.8rem',
                  fontWeight: '600',
                  marginBottom: '30px',
                  color: '#1d1d1f',
                  textAlign: 'center'
                }}>
                  {group.title}
                </h3> */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
                  // gridTemplateColumns: 'repeat(2, 1fr)',
                  gap: '10px'
                }}>
                  {group.videometas.map((videometa) => (
                    <VideoCard
                      videometa={videometa}
                      liquidGlassCard={liquidGlassCard}
                      liquidGlassCardHover={liquidGlassCardHover}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 导航标签 */}
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '20px',
          marginTop: '30px'
        }}>
          {galleryGroups.map((group, index) => (
            <button
              key={index}
              onClick={() => setCurrentGroupIndex(index)}
              style={{
                padding: '10px 24px',
                borderRadius: '50px',
                background: currentGroupIndex === index ? '#1d1d1f' : 'rgba(29, 29, 31, 0.1)',
                color: currentGroupIndex === index ? '#fff' : '#1d1d1f',
                border: 'none',
                fontSize: '1rem',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                boxShadow: currentGroupIndex === index ? '0 4px 12px rgba(0, 0, 0, 0.1)' : 'none'
              }}
              onMouseEnter={(e) => {
                if (currentGroupIndex !== index) {
                  e.currentTarget.style.background = 'rgba(29, 29, 31, 0.2)';
                }
              }}
              onMouseLeave={(e) => {
                if (currentGroupIndex !== index) {
                  e.currentTarget.style.background = 'rgba(29, 29, 31, 0.1)';
                }
              }}
            >
              {group.title}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HSRecurVideoGallerySection2;