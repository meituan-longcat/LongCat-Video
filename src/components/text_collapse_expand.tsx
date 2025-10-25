import React, { useState, useRef, useEffect } from 'react';

interface TextCollapseExpandProps {
  title: string;
  prompt: string;
}


const TextCollapseExpand = ({ title, prompt }: TextCollapseExpandProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isOverflowing, setIsOverflowing] = useState(false);
  const textRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (textRef.current) {
      const lineHeight = parseInt(getComputedStyle(textRef.current).lineHeight);
      const maxHeight = lineHeight * 2;
      const actualHeight = textRef.current.scrollHeight;
      
      // 添加容差范围，解决浏览器渲染差异
      const tolerance = 5; // 2px 容差
      setIsOverflowing(actualHeight > 0 && actualHeight > maxHeight + tolerance);
    }
  }, [prompt]);

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div style={{ 
      padding: '15px', 
      // maxWidth: '600px', 
      margin: '0 auto',
      backgroundColor: '#fff',
      // borderRadius: '12px',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* <h3 style={{ 
        fontWeight: '700', 
        marginBottom: '12px', 
        color: '#1d1d1f', 
        fontSize: '18px',
        position: 'relative',
        paddingBottom: '8px'
      }}>
        {title}
        <div style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          width: '40px',
          height: '3px',
          background: 'linear-gradient(90deg, #FFD700, #FFA500)',
          borderRadius: '2px'
        }}></div>
      </h3> */}
      
      <div style={{ position: 'relative', paddingRight: isOverflowing ? '10px' : '0' }}>
        <p 
          ref={textRef}
          style={{ 
            fontSize: '14px', 
            color: '#5a5a5f', 
            lineHeight: '1.6',
            marginBottom: isOverflowing ? '10px' : '0',
            overflow: 'hidden',
            display: '-webkit-box',
            WebkitBoxOrient: 'vertical',
            WebkitLineClamp: isExpanded ? 'unset' : '2',
            transition: 'all 0.3s ease',
            paddingRight: isOverflowing ? '10px' : '0'
          }}
        >
          {prompt}
        </p>
        
        {isOverflowing && (
          <button
            onClick={toggleExpand}
            style={{
              position: 'absolute',
              bottom: '0',
              right: '0',
              width: '15px',
              height: '15px',
              borderRadius: '50%',
              background: isExpanded ? '#FFF9C4' : 'linear-gradient(135deg, #FFD700, #FFA500)',
              border: 'none',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              zIndex: 10,
              transform: 'translateY(50%)'
            }}
            aria-label={isExpanded ? "折叠内容" : "展开内容"}
          >
            {isExpanded ? (
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                <path d="M3 8H13" stroke="#FF8C00" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            ) : (
              <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                <path d="M8 3V13M3 8H13" stroke="white" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            )}
          </button>
        )}
      </div>
    </div>
  );
};

export default TextCollapseExpand;