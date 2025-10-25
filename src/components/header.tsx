import React, { useCallback } from 'react';

interface HeaderProps {
  tabs?: string[];
}


const Header: React.FC<HeaderProps> = ({ 
  tabs = ['Overview', 'Capabilities', 'Features', 'Platforms', 'FAQ', 'Comparison']
}) => {

  // Apple-style smooth scroll with momentum
  const smoothScrollTo = useCallback((elementId: string) => {
    const element = document.getElementById(elementId)
    if (!element) return

    const startPosition = window.pageYOffset
    const targetPosition = element.offsetTop - 80 // Account for sticky navigation
    const distance = targetPosition - startPosition
    const duration = Math.min(Math.abs(distance) * 1.5, 100) // Dynamic duration with max cap, slightly longer for smoothness

    let startTime: number | null = null

    // Apple's signature easing function with enhanced momentum
    const appleEasing = (t: number): number => {
      // Custom bezier curve that mimics Apple's spring physics
      if (t < 0.5) {
        return 4 * t * t * t
      }
      const f = 2 * t - 2
      return 1 + f * f * f / 2
    }

    const animation = (currentTime: number) => {
      if (startTime === null) startTime = currentTime
      const timeElapsed = currentTime - startTime
      const progress = Math.min(timeElapsed / duration, 1)

      const ease = appleEasing(progress)
      const currentPosition = startPosition + (distance * ease)

      window.scrollTo(0, currentPosition)

      if (progress < 1) {
        requestAnimationFrame(animation)
      }
    }

    requestAnimationFrame(animation)
  }, [])

  const handleTabClick = (e: React.MouseEvent<HTMLButtonElement>, item: string) => {
    // Add quick visual feedback like Apple buttons
    const target = e.currentTarget;
    if (target) {
      target.style.transform = 'scale(0.96)';
      target.style.background = 'rgba(255, 208, 0, 0.15)';

      setTimeout(() => {
        if (target) {
          target.style.transform = 'scale(1)';
          target.style.background = 'rgba(255, 208, 0, 0.08)';
        }
      }, 50);
    }

    smoothScrollTo(item.toLowerCase())
  };

  return (
    <div style={{ 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'space-between', 
      position: 'relative', 
      zIndex: 10 
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
        {/* logo */}
        {/* <div style={{
          width: '48px',
          height: '48px',
          background: 'linear-gradient(135deg, #FFD000 0%, #F0A500 100%)',
          borderRadius: '16px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: '700',
          fontSize: '18px',
          color: '#000000',
          boxShadow: '0 2px 8px rgba(255, 208, 0, 0.2), 0 1px 4px rgba(0, 0, 0, 0.08)',
          border: '0.5px solid rgba(255, 255, 255, 0.3)'
        }}>
          M
        </div> */}
        <img src="assets/icos/longcat_logo.svg" width="250"></img>
        {/* name */}
        {/* <span style={{ fontSize: '28px', fontWeight: '700', color: '#1d1d1f' }}>LongCat-Video</span> */}
      </div>
      {/* tab */}
      <div style={{ display: 'flex', gap: '40px' }}>
        {tabs.map((item) => (
          <button
            key={item}
            onClick={(e) => handleTabClick(e, item)}
            style={{
              color: '#515154',
              textDecoration: 'none',
              fontWeight: '500',
              transition: 'all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
              position: 'relative',
              padding: '10px 16px',
              borderRadius: '12px',
              fontSize: '15px',
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              WebkitTapHighlightColor: 'transparent' // Remove iOS tap highlight
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = '#FFD000';
              e.currentTarget.style.background = 'rgba(255, 208, 0, 0.08)';
              e.currentTarget.style.transform = 'scale(1.02)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = '#515154';
              e.currentTarget.style.background = 'transparent';
              e.currentTarget.style.transform = 'scale(1)';
            }}
          >
            {item}
          </button>
        ))}
      </div>
    </div>
  );
};

export default Header;