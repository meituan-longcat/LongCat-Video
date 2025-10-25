import React from 'react';

interface FeatureItem {
  value: string;
  label: string;
}

interface CapabilitiesSectionProps {
  sectionId: string;
  sectionStyle: React.CSSProperties;
  containerStyle: React.CSSProperties;
  liquidGlassCard: React.CSSProperties;
  liquidGlassCardHover: React.CSSProperties;
}

const CapabilitiesSection: React.FC<CapabilitiesSectionProps> = ({
  sectionId,
  sectionStyle,
  containerStyle,
  liquidGlassCard,
  liquidGlassCardHover
}) => {
  const capabilities = [
    {
        icon: '⚡',
        title: 'Ultra-Realistic Physics',
        description: 'LongCat-Video accurately simulates object interactions, fluid dynamics, and natural motion patterns, including handling extreme physics such as acrobatics.'
    },
    {
        icon: '🎬',
        title: 'Cinematic Quality',
        description: 'Professional-grade visual fidelity with perfect lighting and composition, suitable for film, marketing, education, and social media content creation.'
    },
    {
        icon: '👤',
        title: 'Character Consistency',
        description: 'Advanced facial recognition and body tracking maintain consistent character appearances across frames, addressing a common AI video challenge.'
    }
  ]
  const coreFeatures = [
    'Text-to-video generation',
    'Image-to-video conversion',
    'Advanced camera controls (pan-down, dolly zoom)',
    'Frame-perfect motion precision',
    'Multilingual support (English, Chinese, +)'
  ];

  const techSpecs: FeatureItem[] = [
    { value: '2.5x', label: 'Faster Training & Inference' },
    { value: '3x', label: 'Larger Parameters' },
    { value: '4x', label: 'More Training Data' }
  ];

  return (
    <section id={sectionId} style={{ ...sectionStyle, position: 'relative', zIndex: 5 }}>
      <div style={containerStyle}>
        <div style={{ textAlign: 'center', marginBottom: '60px' }}>
          <h2 style={{ fontSize: '3rem', fontWeight: 'bold', marginBottom: '20px', color: '#1d1d1f' }}>
            LongCat-Video Advanced Capabilities
          </h2>
        </div>

        {/* Capabilities Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: '40px' }}>
          {capabilities.map((capability) => (
            <div 
              key={capability.title}
              style={{
                ...liquidGlassCard,
                padding: '50px',
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
              <div style={{ fontSize: '5rem', marginBottom: '25px', filter: 'none' }}>
                {capability.icon}
              </div>
              <h3 style={{ fontSize: '1.6rem', fontWeight: 'bold', marginBottom: '20px', color: '#1d1d1f' }}>
                {capability.title}
              </h3>
              <p style={{ color: '#86868b', lineHeight: '1.6', fontSize: '15px' }}>
                {capability.description}
              </p>
            </div>
          ))}
        </div>

        {/* Core Features Section */}
        <div style={{ marginTop: '80px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '60px' }}>
          {/* Features List */}
          <div style={{ ...liquidGlassCard, padding: '40px' }}>
            <h3 style={{ fontSize: '1.6rem', fontWeight: 'bold', marginBottom: '30px', color: '#FFD000' }}>
              LongCat-Video Core Features
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
              {coreFeatures.map((feature) => (
                <div key={feature} style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                  <span style={{ color: '#FFD000', fontSize: '20px', fontWeight: 'bold' }}>✓</span>
                  <span style={{ color: '#495057' }}>{feature}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Technical Specs */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.15)',
            backdropFilter: 'blur(12px)',
            WebkitBackdropFilter: 'blur(12px)',
            border: '0.5px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '24px',
            padding: '40px',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.08), 0 1px 4px rgba(0, 0, 0, 0.04)'
          }}>
            <h3 style={{ fontSize: '1.6rem', fontWeight: 'bold', marginBottom: '30px', color: '#FFD000' }}>
              Technical Architecture
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '25px' }}>
              {techSpecs.map((item) => (
                <div key={item.value} style={{ textAlign: 'center' }}>
                  <div style={{
                    fontSize: '2.5rem',
                    fontWeight: 'bold',
                    background: 'linear-gradient(135deg, #FFD000, #FFE033)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    filter: 'none'
                  }}>
                    {item.value}
                  </div>
                  <div style={{ 
                    fontSize: '14px', 
                    color: '#86868b', 
                    textTransform: 'uppercase', 
                    letterSpacing: '1px' 
                  }}>
                    {item.label}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default CapabilitiesSection;