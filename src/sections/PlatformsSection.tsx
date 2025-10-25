import React from 'react';

interface StepItem {
  step: string;
  title: string;
  description: string;
}

interface PlatformsSectionProps {
  sectionId: string;
  sectionStyle: React.CSSProperties;
  containerStyle: React.CSSProperties;
  liquidGlassCard: React.CSSProperties;
  liquidGlassCardHover: React.CSSProperties;
}

const PlatformsSection: React.FC<PlatformsSectionProps> = ({
  sectionId,
  sectionStyle,
  containerStyle,
  liquidGlassCard,
  liquidGlassCardHover
}) => {
  const platforms = [
    {
        icon: '🌐',
        title: 'Official Website',
        description: 'Explore LongCat-Video capabilities directly through the official platform.',
        link: 'Visit LongCat-Video AI',
        url: 'https://LongCat-Video.video/create'
    },
    {
        icon: '⚙️',
        title: 'BasedLabs',
        description: 'Access LongCat-Video through BasedLabs with text-to-video and image-to-video features. Supports 70+ languages.',
        details: 'Cost: 300 credits per video',
        link: 'Try on BasedLabs',
        url: 'https://www.basedlabs.ai/tools/LongCat-Video-ai-video-generator'
    },
    {
        icon: '⚡',
        title: 'fal\'s API Platform',
        description: 'Developer-friendly API access for integrating LongCat-Video into applications.',
        details: '$0.28 per video - Enhanced cost-efficiency for developers',
        link: 'API Access'
    }
  ]
  const howToSteps: StepItem[] = [
    { 
      step: '1', 
      title: 'Launch Editor', 
      description: 'Click "Generate" on the tool page to launch the editor' 
    },
    { 
      step: '2', 
      title: 'Select Model', 
      description: 'Choose "Text to Video" and select " LongCat-Video AI"' 
    },
    { 
      step: '3', 
      title: 'Generate', 
      description: 'Enter your prompt or upload image, then click "Generate"' 
    }
  ];

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
            LongCat-Video Platform Accessibility
          </h2>
        </div>

        {/* Platforms Grid */}
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', 
          gap: '40px' 
        }}>
          {platforms.map((platform) => (
            <div 
              key={platform.title}
              style={{
                ...liquidGlassCard,
                padding: '40px',
                textAlign: 'center',
                cursor: 'pointer'
              }}
              onMouseEnter={(e) => {
                Object.assign(e.currentTarget.style, liquidGlassCardHover);
              }}
              onMouseLeave={(e) => {
                Object.assign(e.currentTarget.style, liquidGlassCard);
              }}
            >
              <div style={{ 
                fontSize: '4rem', 
                marginBottom: '20px', 
                filter: 'none' 
              }}>
                {platform.icon}
              </div>
              <h3 style={{ 
                fontSize: '1.6rem', 
                fontWeight: 'bold', 
                marginBottom: '20px', 
                color: '#1d1d1f' 
              }}>
                {platform.title}
              </h3>
              <p style={{ 
                color: '#86868b', 
                lineHeight: '1.6', 
                marginBottom: '20px', 
                fontSize: '15px' 
              }}>
                {platform.description}
              </p>
              {platform.details && (
                <p style={{
                  background: 'linear-gradient(135deg, #FFD000, #FFE033)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  fontSize: '14px',
                  marginBottom: '20px',
                  fontWeight: '700'
                }}>
                  {platform.details}
                </p>
              )}
              <button 
                style={{
                  background: 'transparent',
                  border: '2px solid #FFD000',
                  color: '#FFD000',
                  padding: '15px 30px',
                  borderRadius: '12px',
                  fontWeight: '700',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  textTransform: 'uppercase',
                  letterSpacing: '1px'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = '#FFD000';
                  e.currentTarget.style.color = '#000000';
                  e.currentTarget.style.boxShadow = '0 8px 25px rgba(255, 208, 0, 0.4)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.color = '#FFD000';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                {platform.link}
              </button>
            </div>
          ))}
        </div>

        {/* How to Use Guide */}
        <div style={{ marginTop: '80px' }}>
          <h3 style={{ 
            fontSize: '2.5rem', 
            fontWeight: 'bold', 
            textAlign: 'center', 
            marginBottom: '60px', 
            color: '#1d1d1f' 
          }}>
            How to Use LongCat-Video on BasedLabs
          </h3>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', 
            gap: '40px' 
          }}>
            {howToSteps.map((item) => (
              <div 
                key={item.step}
                style={{
                  background: 'rgba(255, 255, 255, 0.15)',
                  backdropFilter: 'blur(12px)',
                  WebkitBackdropFilter: 'blur(12px)',
                  border: '0.5px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '24px',
                  padding: '40px',
                  textAlign: 'center',
                  boxShadow: '0 4px 16px rgba(0, 0, 0, 0.08), 0 1px 4px rgba(0, 0, 0, 0.04)'
                }}
              >
                <div style={{
                  width: '56px',
                  height: '56px',
                  background: 'linear-gradient(135deg, #FFD000, #F0A500)',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 24px',
                  fontSize: '20px',
                  fontWeight: '700',
                  color: '#000000',
                  boxShadow: '0 4px 16px rgba(255, 208, 0, 0.2), 0 2px 8px rgba(0, 0, 0, 0.08)'
                }}>
                  {item.step}
                </div>
                <h4 style={{ 
                  fontSize: '1.4rem', 
                  fontWeight: 'bold', 
                  marginBottom: '15px', 
                  color: '#1d1d1f' 
                }}>
                  {item.title}
                </h4>
                <p style={{ 
                  color: '#86868b', 
                  fontSize: '14px', 
                  lineHeight: '1.5' 
                }}>
                  {item.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default PlatformsSection;