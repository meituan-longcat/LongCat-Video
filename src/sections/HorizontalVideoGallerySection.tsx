import React, { useRef, useState, useCallback } from 'react';
import { VideoMeta } from '../data/videometa';

interface HSVideoGallerySectionProps {
  sectionId: string;
  sectionStyle: React.CSSProperties;
  containerStyle: React.CSSProperties;
  liquidGlassCard: React.CSSProperties;
  liquidGlassCardHover: React.CSSProperties;
  videometas: VideoMeta[];
}

const HSVideoGallerySection: React.FC<HSVideoGallerySectionProps> = ({
  sectionId,
  sectionStyle,
  containerStyle,
  liquidGlassCard,
  liquidGlassCardHover,
  videometas
}) => {
  const mainGalleryRef = useRef<HTMLDivElement>(null);
  const [mainGalleryIndex, setMainGalleryIndex] = useState(0);

  const buttonStyle = {
    position: 'absolute' as const,
    top: '50%',
    transform: 'translateY(-50%)',
    zIndex: 10,
    width: '48px',
    height: '48px',
    borderRadius: '50%',
    background: 'rgba(255, 255, 255, 0.9)',
    backdropFilter: 'blur(12px)',
    WebkitBackdropFilter: 'blur(12px)',
    border: '0.5px solid rgba(255, 255, 255, 0.3)',
    boxShadow: '0 4px 16px rgba(0, 0, 0, 0.1)',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.3s ease',
  }


  // Gallery navigation functions
  const scrollGallery = useCallback((galleryRef: React.RefObject<HTMLDivElement>, direction: 'left' | 'right', galleryType: 'main' | 'featured') => {
    if (!galleryRef.current) return

    const container = galleryRef.current
    const containerWidth = container.clientWidth

    // 计算每个视频卡片的实际宽度（70% + gap）
    const cardWidth = containerWidth * 0.7
    const gap = galleryType === 'main' ? 20 : 24
    const itemStep = cardWidth + gap

    // 获取当前滚动位置
    const currentScroll = container.scrollLeft

    // 计算当前视频索引（基于实际滚动位置）
    const currentIndex = Math.round(currentScroll / itemStep)

    // 确定目标索引
    const maxIndex = galleryType === 'main' ? videometas.length - 1 : 3
    let targetIndex = direction === 'left' ? currentIndex - 1 : currentIndex + 1
    targetIndex = Math.max(0, Math.min(targetIndex, maxIndex))

    // 计算目标滚动位置
    const targetScroll = targetIndex * itemStep

    container.scrollTo({
      left: targetScroll,
      behavior: 'smooth'
    })
  }, [])

  const updateGalleryIndex = useCallback((galleryRef: React.RefObject<HTMLDivElement>, setIndex: (index: number) => void, galleryType: 'main' | 'featured') => {
    if (!galleryRef.current) return

    const container = galleryRef.current
    const scrollLeft = container.scrollLeft
    const scrollWidth = container.scrollWidth
    const clientWidth = container.clientWidth
    const maxScroll = scrollWidth - clientWidth

    const maxIndex = galleryType === 'main' ? videometas.length - 1 : 3

    // 如果滚动到最后，显示最后一个指示器
    if (scrollLeft >= maxScroll - 20) {
      setIndex(maxIndex)
      return
    }

    // 计算每个视频的step
    const cardWidth = clientWidth * 0.7
    const gap = galleryType === 'main' ? 20 : 24
    const itemStep = cardWidth + gap

    // 简单直接的索引计算
    const index = Math.round(scrollLeft / itemStep)
    setIndex(Math.max(0, Math.min(index, maxIndex)))
  }, [])

  return (
    <section id={sectionId} style={{ ...sectionStyle, position: 'relative', zIndex: 5 }}>
      <style>
        {`
          .horizontal-scroll::-webkit-scrollbar {
            display: none;
          }
          .horizontal-scroll {
            scrollbar-width: none;
            -ms-overflow-style: none;
            scroll-snap-type: x mandatory;
          }
          .horizontal-scroll > * {
            scroll-snap-align: center;
            scroll-snap-stop: always;
          }
        `}
      </style>
      
      <div style={containerStyle}>
        <div style={{ textAlign: 'center', marginBottom: '60px' }}>
          <h2 style={{ 
            fontSize: '3rem', 
            fontWeight: 'bold', 
            marginBottom: '20px', 
            color: '#1d1d1f' 
          }}>
            LongCat-Video Video Gallery - See the Magic in Action
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

        {/* Horizontal Scrollable Gallery */}
        <div style={{ 
            position: 'relative', 
            // overflow: 'hidden' 
        }}>
          {/* Left Arrow */}
          <button
            onClick={() => scrollGallery(mainGalleryRef, 'left', 'main')}
            style={{
              ...buttonStyle,
              left: '-60px',
              opacity: mainGalleryIndex === 0 ? 0.5 : 1
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

          {/* Right Arrow */}
          <button
            onClick={() => scrollGallery(mainGalleryRef, 'right', 'main')}
            style={{
              ...buttonStyle,
              right: '-60px',
              opacity: mainGalleryIndex >= videometas.length - 1 ? 0.5 : 1
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

          <div
            ref={mainGalleryRef}
            className="horizontal-scroll"
            style={{
              display: 'flex',
              gap: '20px',
              overflowX: 'auto',
              scrollBehavior: 'smooth',
              paddingBottom: '20px',
              WebkitOverflowScrolling: 'touch',
              scrollPadding: '0 20px'
            }}
            onScroll={() => updateGalleryIndex(mainGalleryRef, setMainGalleryIndex, 'main')}
          >
            {videometas.map((videometa) => (
              <div 
                // key={videometa.id} 
                style={{
                  width: '70%',
                  minWidth: '70%',
                  maxWidth: '500px',
                  flexShrink: 0,
                  ...liquidGlassCard,
                  overflow: 'hidden',
                  cursor: 'pointer',
                  padding: '0',
                  scrollSnapAlign: 'center', // 居中对齐主视频
                  transition: 'all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94)'
                }}
                onMouseEnter={(e) => {
                  Object.assign(e.currentTarget.style, {
                    ...liquidGlassCardHover,
                    transform: 'translateY(-12px) scale(1.05)',
                    zIndex: 10
                  });
                }}
                onMouseLeave={(e) => {
                  Object.assign(e.currentTarget.style, {
                    ...liquidGlassCard,
                    transform: 'translateY(0) scale(1)',
                    zIndex: 1
                  });
                }}
              >
                {/* video */}
                <div style={{ position: 'relative' }}>
                  <video
                    src={videometa.url}
                    style={{
                      width: '100%',
                      height: '500px',
                      objectFit: 'cover', // 占满容器
                    }}
                    muted
                    loop
                    // onMouseEnter={(e) => {
                    //   e.currentTarget.play();
                    // }}
                    // onMouseLeave={(e) => {
                    //   e.currentTarget.pause();
                    // }}
                  />
                  <div style={{
                    position: 'absolute',
                    top: '16px',
                    left: '16px',
                    background: 'linear-gradient(135deg, #FFD000 0%, #F0A500 100%)',
                    color: '#000000',
                    padding: '8px 16px',
                    borderRadius: '16px',
                    fontSize: '12px',
                    fontWeight: '700',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    boxShadow: '0 4px 16px rgba(255, 208, 0, 0.3), 0 2px 8px rgba(0, 0, 0, 0.1)',
                    backdropFilter: 'blur(8px)',
                    WebkitBackdropFilter: 'blur(8px)'
                  }}>
                    {videometa.label}
                  </div>

                  {/* Play button overlay */}
                  {/* <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    width: '60px',
                    height: '60px',
                    background: 'rgba(255, 255, 255, 0.9)',
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    opacity: 0.8,
                    transition: 'all 0.3s ease',
                    backdropFilter: 'blur(10px)',
                    WebkitBackdropFilter: 'blur(10px)',
                    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)'
                  }}>
                    <div style={{
                      width: 0,
                      height: 0,
                      borderLeft: '16px solid #1d1d1f',
                      borderTop: '10px solid transparent',
                      borderBottom: '10px solid transparent',
                      marginLeft: '4px'
                    }} />
                  </div> */}
                </div>
                
                {/* prompt */}
                <div style={{ padding: '24px' }}>
                  {/* 标题 */}
                  <h3 style={{
                    fontWeight: '700',
                    marginBottom: '12px',
                    color: '#1d1d1f',
                    fontSize: '18px',
                    letterSpacing: '-0.02em'
                  }}>
                    {videometa.title}
                  </h3>

                  <p style={{
                    fontSize: '14px',
                    color: '#86868b',
                    lineHeight: '1.6',
                    marginBottom: '1px'
                  }}>
                    {videometa.prompt}
                  </p>

                  {/* Watch button */}
                  {/* <button style={{
                    background: 'rgba(255, 208, 0, 0.1)',
                    border: '1px solid rgba(255, 208, 0, 0.3)',
                    color: '#FFD000',
                    padding: '10px 20px',
                    borderRadius: '20px',
                    fontSize: '13px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    textTransform: 'none',
                    letterSpacing: '0.2px'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'rgba(255, 208, 0, 0.2)';
                    e.currentTarget.style.transform = 'scale(1.05)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'rgba(255, 208, 0, 0.1)';
                    e.currentTarget.style.transform = 'scale(1)';
                  }}
                  >
                    Watch Video
                  </button> */}
                </div>
              </div>
            ))}
          </div>

          {/* Active Scroll indicators */}
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '8px',
            marginTop: '32px'
          }}>
            {videometas.map((_, index) => (
              <button
                key={`main-gallery-indicator-${index}`}
                onClick={() => {
                  if (mainGalleryRef.current) {
                    const containerWidth = mainGalleryRef.current.clientWidth;
                    const cardWidth = containerWidth * 0.7;
                    const gap = 20;
                    const scrollPosition = index * (cardWidth + gap);
                    
                    mainGalleryRef.current.scrollTo({
                      left: scrollPosition,
                      behavior: 'smooth'
                    });
                  }
                }}
                style={{
                  width: mainGalleryIndex === index ? '24px' : '8px',
                  height: '8px',
                  borderRadius: '4px',
                  background: mainGalleryIndex === index ? '#FFD000' : 'rgba(255, 208, 0, 0.3)',
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94)'
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default HSVideoGallerySection;