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

const HSRecurVideoGallerySection5: React.FC<VideoGallerySectionProps> = ({
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
    marginBottom: '30px', 
    color: '#1d1d1f' 
  }}>
    Method
  </h2>
  
  <div style={{
    maxWidth: '1400px',
    margin: '0 auto',
    textAlign: 'left',
    padding: '0 20px'
  }}>
    <ul style={{
      listStyleType: 'none',
      padding: 0,
      margin: 0,
      display: 'flex',
      flexDirection: 'column',
      gap: '25px'
    }}>
          {/* 特性 1 */}
          <li style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: '12px'
          }}>
            <span style={{
              fontSize: '24px',
              color: '#FF8A00',
              lineHeight: '1'
            }}>🌟</span>
            <div>
              <h3 style={{
                fontSize: '1.25rem',
                fontWeight: '600',
                color: '#1d1d1f',
                margin: '0 0 8px 0'
              }}>
                Unified Architecture for Multiple Tasks
              </h3>
              <p style={{
                color: '#86868b',
                fontSize: '1rem',
                lineHeight: '1.6',
                margin: 0
              }}>
                LongCat-Video unifies Text-to-Video, Image-to-Video, and Video-Continuation tasks within a single framework, delivering strong performance across all tasks with one model.
              </p>
            </div>
          </li>

          {/* 特性 2 */}
          <li style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: '12px'
          }}>
            <span style={{
              fontSize: '24px',
              color: '#FF8A00',
              lineHeight: '1'
            }}>🌟</span>
            <div>
              <h3 style={{
                fontSize: '1.25rem',
                fontWeight: '600',
                color: '#1d1d1f',
                margin: '0 0 8px 0'
              }}>
                Long Video Generation
              </h3>
              <p style={{
                color: '#86868b',
                fontSize: '1rem',
                lineHeight: '1.6',
                margin: 0
              }}>
                Pretrained on Video-Continuation tasks to produce minutes-long videos without color drifting or quality degradation.
              </p>
            </div>
          </li>

          {/* 特性 3 */}
          <li style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: '12px'
          }}>
            <span style={{
              fontSize: '24px',
              color: '#FF8A00',
              lineHeight: '1'
            }}>🌟</span>
            <div>
              <h3 style={{
                fontSize: '1.25rem',
                fontWeight: '600',
                color: '#1d1d1f',
                margin: '0 0 8px 0'
              }}>
                Efficient Inference
              </h3>
              <p style={{
                color: '#86868b',
                fontSize: '1rem',
                lineHeight: '1.6',
                margin: 0
              }}>
                Generates 720p 30fps videos in minutes via Coarse-to-Fine generation and Block Sparse Attention for high-resolution efficiency.
              </p>
            </div>
          </li>

          {/* 特性 4 */}
          <li style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: '12px'
          }}>
            <span style={{
              fontSize: '24px',
              color: '#FF8A00',
              lineHeight: '1'
            }}>🌟</span>
            <div>
              <h3 style={{
                fontSize: '1.25rem',
                fontWeight: '600',
                color: '#1d1d1f',
                margin: '0 0 8px 0'
              }}>
                Strong Performance with Multi-Reward RLHF
              </h3>
              <p style={{
                color: '#86868b',
                fontSize: '1rem',
                lineHeight: '1.6',
                margin: 0
              }}>
                Powered by GRPO, outperforming benchmarks and matching leading open-source/commercial solutions.
              </p>
            </div>
          </li>
        </ul>
      </div>
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
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(1, 1fr)',
                  gap: '10px'
                }}>
                  {group.videometas.map((videometa) => (
                    <div 
                    style={{
                      ...liquidGlassCard,
                      position: 'relative',
                      overflow: 'hidden',
                      transition: 'all 0.3s ease',
                      padding: '20px',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center'
                    }}
                    onMouseEnter={(e) => {
                      Object.assign(e.currentTarget.style, liquidGlassCardHover);
                    }}
                    onMouseLeave={(e) => {
                      Object.assign(e.currentTarget.style, liquidGlassCard);
                    }}
                  >
                    {/* 图片展示 */}
                    <div style={{
                      width: '100%',
                      height: '600px',
                      borderRadius: '12px',
                      overflow: 'hidden',
                      marginBottom: '16px',
                      position: 'relative',
                    }}>
                      <img 
                        src={videometa.url} 
                        // alt={videometa.title}
                        style={{
                          width: '70%',
                          height: '100%',
                          objectFit: 'cover',
                          transition: 'transform 0.5s ease',
                          display: 'block', // 关键：将img转为块级元素
                          margin: '0 auto'  // 关键：水平居中
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.transform = 'scale(1.05)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.transform = 'scale(1)';
                        }}
                      />
                    </div>
                    
                    {/* 图注/标题 */}
                    <div style={{
                      textAlign: 'center',
                      padding: '0 10px'
                    }}>
                      {/* <h3 style={{
                        fontSize: '1.25rem',
                        fontWeight: '600',
                        marginBottom: '8px',
                        color: '#1d1d1f'
                      }}>
                        {videometa.title}
                      </h3> */}
                      {/* <p style={{
                        color: '#86868b',
                        fontSize: '0.95rem',
                        lineHeight: '1.5'
                      }}>
                        {videometa.prompt}
                      </p> */}
                    </div>
                  </div>
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

export default HSRecurVideoGallerySection5;