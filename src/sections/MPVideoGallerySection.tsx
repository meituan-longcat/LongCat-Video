import React from 'react';
import MPVideoCard from '../components/video_card_mp';
import { VideoMetaMP } from '../data/videometa';

interface VideoGallerySectionProps {
  sectionId: string;
  sectionStyle: React.CSSProperties;
  containerStyle: React.CSSProperties;
  liquidGlassCard: React.CSSProperties;
  liquidGlassCardHover: React.CSSProperties;
  videometas: VideoMetaMP[];
}

const MPVideoGallerySection: React.FC<VideoGallerySectionProps> = ({
  sectionId,
  sectionStyle,
  containerStyle,
  liquidGlassCard,
  liquidGlassCardHover,
  videometas
}) => {
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
            World Model Gallery
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

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(600px, 1fr))',
          gap: '30px'
        }}>
          {videometas.map((videometa) => (
            <MPVideoCard
              videometa={videometa}
              liquidGlassCard={liquidGlassCard}
              liquidGlassCardHover={liquidGlassCardHover}
            />
          ))}
        </div>
      </div>
    </section>
  );
};

export default MPVideoGallerySection;