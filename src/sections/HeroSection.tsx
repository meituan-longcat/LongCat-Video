import React, { useEffect, useRef } from 'react';

const HeroSection = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  
  // 确保视频自动播放
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.play().catch(error => {
        console.log('Autoplay prevented:', error);
      });
    }
  }, []);

  // 容器样式
  const containerStyle: React.CSSProperties = {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '0 20px',
    position: 'relative',
    zIndex: 10,
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    textAlign: 'center',
  };

  // 渐变文字样式
  const gradientTextStyle: React.CSSProperties = {
    fontSize: '5.5rem',
    fontWeight: 'bold',
    background: 'linear-gradient(135deg, #FFD000 0%, #FF7A00 50%, #FF006E 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    marginBottom: '30px',
    lineHeight: '1.5',
    letterSpacing: '-0.03em',
    textShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
  };

  return (
    <section style={{
      position: 'relative',
      height: '100vh',
      minHeight: '700px',
      overflow: 'hidden',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    }}>
      {/* 视频背景 */}
      <video
        ref={videoRef}
        autoPlay
        muted
        loop
        playsInline
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          minWidth: '100%',
          minHeight: '100%',
          width: 'auto',
          height: 'auto',
          transform: 'translate(-50%, -50%)',
          zIndex: 1,
          objectFit: 'cover',
        }}
      >
        <source src="assets/videos/teaser-30.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>
      
      {/* 深色遮罩层 */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        background: 'linear-gradient(to bottom, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.3) 100%)',
        zIndex: 2,
      }} />
      
      {/* 内容容器 */}
      <div style={containerStyle}>
        <div style={{ position: 'relative', zIndex: 10 }}>
          <h1 style={
            gradientTextStyle
            }>LongCat-Video</h1>
          <h2 style={{ 
              fontSize: '2.2rem', 
              color: '#ffffff', 
              marginBottom: '32px', 
              fontWeight: '600', 
              letterSpacing: '-0.02em',
              textShadow: '0 2px 10px rgba(0,0,0,0.3)'
          }}>
               A Unified Foundational Video Generation Model 
          </h2>
          <p style={{ 
              fontSize: '20px', 
              color: 'rgba(255,255,255,0.9)', 
              marginBottom: '60px', 
              maxWidth: '800px', 
              margin: '0 auto 60px', 
              lineHeight: '1.6',
              textShadow: '0 1px 4px rgba(0,0,0,0.3)'
          }}>
              LongCat-Video delivers strong performance across Text-to-Video, Image-to-Video, and Video-Continuation generation tasks. It particularly excels in efficient and high-quality long video generation, representing our first step toward world models.
          </p>
          
          {/* 行动按钮 */}
          {/* 行动按钮 */}
          <div style={{ display: 'flex', gap: '20px', justifyContent: 'center' }}>
            <button 
              onClick={() => window.open('https://github.com/meituan-longcat/LongCat-Video', '_blank')}
              style={{
              background: 'linear-gradient(135deg, #FF006E 0%, #FF7A00 100%)',
              color: 'white',
              border: 'none',
              padding: '10px 30px 10px 20px',
              fontSize: '1.1rem',
              fontWeight: '600',
              borderRadius: '50px',
              cursor: 'pointer',
              boxShadow: '0 8px 20px rgba(255, 0, 110, 0.3)',
              transition: 'all 0.3s ease',
              zIndex: 10,
              position: 'relative',
              display: 'inline-flex', // 使用 inline-flex 而不是 flex
              alignItems: 'center',
              justifyContent: 'center', // 确保内容居中
              gap: '10px',
              minWidth: '140px',
              lineHeight: '1.5', // 确保文字行高一致
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-3px)';
              e.currentTarget.style.boxShadow = '0 12px 25px rgba(255, 0, 110, 0.4)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 8px 20px rgba(255, 0, 110, 0.3)';
            }}
            >
              <svg 
                height="18" 
                width="18" 
                viewBox="0 0 16 16" 
                fill="white"
                style={{ 
                  display: 'inline-block', // 确保 SVG 是行内块元素
                  verticalAlign: 'middle', // 垂直居中
                  flexShrink: 0 // 防止 SVG 被压缩
                }}
              >
                <path fillRule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
              </svg>
              <span style={{ 
                display: 'inline-block', // 确保文字是行内块元素
                verticalAlign: 'middle', // 垂直居中
              }}>
                Code
              </span>
            </button>
            <button 
              onClick={() => window.open('https://huggingface.co/meituan-longcat/LongCat-Video', '_blank')}
              style={{
              background: 'linear-gradient(135deg, #FF006E 0%, #FF7A00 100%)',
              color: 'white',
              border: 'none',
              padding: '10px 30px 10px 20px',
              fontSize: '1.1rem',
              fontWeight: '600',
              borderRadius: '50px',
              cursor: 'pointer',
              boxShadow: '0 8px 20px rgba(255, 0, 110, 0.3)',
              transition: 'all 0.3s ease',
              zIndex: 10,
              position: 'relative',
              display: 'inline-flex', // 使用 inline-flex 而不是 flex
              alignItems: 'center',
              justifyContent: 'center', // 确保内容居中
              gap: '10px',
              minWidth: '140px',
              lineHeight: '1.5', // 确保文字行高一致
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-3px)';
              e.currentTarget.style.boxShadow = '0 12px 25px rgba(255, 0, 110, 0.4)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 8px 20px rgba(255, 0, 110, 0.3)';
            }}
            >
              <img src="assets/icos/hf-logo.svg" width="30" height="30"></img>
              <span style={{ 
                display: 'inline-block', // 确保文字是行内块元素
                verticalAlign: 'middle', // 垂直居中
              }}>
                Model
              </span>
            </button>
            <button 
              onClick={() => window.open('https://github.com/meituan-longcat/LongCat-Video/blob/main/longcatvideo_tech_report.pdf', '_blank')}
              style={{
              background: 'linear-gradient(135deg, #FF006E 0%, #FF7A00 100%)',
              color: 'white',
              border: 'none',
              padding: '10px 30px 10px 20px',
              fontSize: '1.1rem',
              fontWeight: '600',
              borderRadius: '50px',
              cursor: 'pointer',
              boxShadow: '0 8px 20px rgba(255, 0, 110, 0.3)',
              transition: 'all 0.3s ease',
              zIndex: 10,
              position: 'relative',
              display: 'inline-flex', // 使用 inline-flex 而不是 flex
              alignItems: 'center',
              justifyContent: 'center', // 确保内容居中
              gap: '10px',
              minWidth: '140px',
              lineHeight: '1.5', // 确保文字行高一致
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-3px)';
              e.currentTarget.style.boxShadow = '0 12px 25px rgba(255, 0, 110, 0.4)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 8px 20px rgba(255, 0, 110, 0.3)';
            }}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M14 3H7C5.89543 3 5 3.89543 5 5V19C5 20.1046 5.89543 21 7 21H17C18.1046 21 19 20.1046 19 19V8M14 3L19 8M14 3V8H19" strokeWidth="2"/>
              <path d="M8 12H16M8 16H12" strokeWidth="2" strokeLinecap="round"/>
            </svg>
              <span style={{ 
                display: 'inline-block', // 确保文字是行内块元素
                verticalAlign: 'middle', // 垂直居中
              }}>
                Report
              </span>
            </button>
          </div>


        </div>
      </div>
      
      {/* 向下滚动指示器 */}
      <div style={{
        position: 'absolute',
        bottom: '40px',
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 10,
        animation: 'bounce 2s infinite',
      }}>
        <div style={{
          width: '30px',
          height: '50px',
          borderRadius: '15px',
          border: '2px solid rgba(255,255,255,0.5)',
          position: 'relative',
        }}>
          <div style={{
            width: '6px',
            height: '6px',
            background: 'white',
            borderRadius: '50%',
            position: 'absolute',
            top: '10px',
            left: '50%',
            transform: 'translateX(-50%)',
            animation: 'scrollIndicator 2s infinite',
          }} />
        </div>
      </div>
      
      {/* 动画样式 */}
      <style>{`
        @keyframes bounce {
          0%, 20%, 50%, 80%, 100% {
            transform: translateX(-50%) translateY(0);
          }
          40% {
            transform: translateX(-50%) translateY(-20px);
          }
          60% {
            transform: translateX(-50%) translateY(-10px);
          }
        }
        
        @keyframes scrollIndicator {
          0% {
            opacity: 1;
            top: 10px;
          }
          100% {
            opacity: 0;
            top: 30px;
          }
        }
      `}</style>
    </section>
  );
};

export default HeroSection;