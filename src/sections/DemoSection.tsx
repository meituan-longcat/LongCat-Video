import React from 'react';

interface DemoSectionProps {
  sectionStyle: React.CSSProperties;
  containerStyle: React.CSSProperties;
  liquidGlassCard: React.CSSProperties;
  buttonStyle: React.CSSProperties;
  videoSrc: string;
  videoPoster: string;
}

const DemoSection: React.FC<DemoSectionProps> = ({
  sectionStyle,
  containerStyle,
  liquidGlassCard,
  buttonStyle,
  videoSrc,
  videoPoster
}) => {
  return (
    <section style={{ 
      ...sectionStyle, 
      background: 'rgba(248, 249, 250, 0.8)', 
      backdropFilter: 'blur(8px)', 
      WebkitBackdropFilter: 'blur(8px)', 
      position: 'relative', 
      zIndex: 5 
    }}>
      <div style={containerStyle}>
        <div style={{ textAlign: 'center', marginBottom: '60px' }}>
          <h2 style={{ fontSize: '3rem', fontWeight: 'bold', marginBottom: '20px', color: '#1d1d1f' }}>
            See LongCat-Video in Action
          </h2>
          <p style={{ color: '#86868b', fontSize: '18px' }}>
            Official LongCat-Video demonstration showcasing cinematic quality and realistic physics simulation
          </p>
        </div>
        <div style={{ maxWidth: '900px', margin: '0 auto' }}>
          <div style={{
            ...liquidGlassCard,
            padding: '0',
            overflow: 'hidden',
            position: 'relative'
          }}>
            <video
              src={videoSrc}
              controls
              style={{ width: '100%', height: 'auto', borderRadius: '15px' }}
            //   poster={videoPoster}
            >
              <track kind="captions" src="" label="English captions" />
              Your browser does not support the video tag.
            </video>
            {/* tag */}
            <div style={{
              position: 'absolute',
              top: '25px',
              left: '25px',
              background: 'linear-gradient(135deg, #FFD000 0%, #FFE033 100%)',
              color: '#000000',
              padding: '10px 20px',
              borderRadius: '20px',
              fontSize: '14px',
              fontWeight: '700',
              textTransform: 'uppercase',
              letterSpacing: '1px',
              boxShadow: '0 4px 15px rgba(255, 208, 0, 0.3)'
            }}>
              Intelligence for Everyone
            </div>
          </div>
          {/* button */}
          <div style={{ textAlign: 'center', marginTop: '40px' }}>
            <button 
              style={buttonStyle}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px) scale(1.02)';
                e.currentTarget.style.boxShadow = '0 8px 24px rgba(255, 208, 0, 0.3), 0 4px 12px rgba(0, 0, 0, 0.12)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0) scale(1)';
                e.currentTarget.style.boxShadow = '0 4px 16px rgba(255, 208, 0, 0.2), 0 2px 8px rgba(0, 0, 0, 0.08)';
              }}
            >
              Click to Experience Now
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default DemoSection;