import React from 'react';
import TextCollapseExpand from './text_collapse_expand'
import { VideoMeta } from '../data/videometa';

interface VideoCardProps {
  videometa: VideoMeta;
  liquidGlassCard: React.CSSProperties;
  liquidGlassCardHover: React.CSSProperties;
  height?: string;
}

const VideoCard: React.FC<VideoCardProps> = ({ videometa, liquidGlassCard, liquidGlassCardHover, height='300px' }) => {
  return (
    <div
      // key={videometa.id}
      style={{
        ...liquidGlassCard,
        overflow: 'hidden',
        cursor: 'pointer',
        padding: '0'
      }}
      onMouseEnter={(e) => {
        Object.assign(e.currentTarget.style, liquidGlassCardHover);
      }}
      onMouseLeave={(e) => {
        Object.assign(e.currentTarget.style, liquidGlassCard);
      }}
    >
      <div style={{ position: 'relative' }}>
        <video
          src={videometa.url}
          style={{ 
            width: '100%', 
            // height: '300px', 
            height: height, 
            objectFit: 'cover', 
          }}
          muted
          controls
          loop
        />
        {/* <div style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          background: 'linear-gradient(135deg, #FFD000 0%,rgb(255, 160, 51) 100%)',
          color: '#000000',
          padding: '4px 8px',
          borderRadius: '12px',
          fontSize: '10px',
          fontWeight: '700',
          textTransform: 'uppercase',
          letterSpacing: '1px',
          boxShadow: '0 4px 15px rgba(255, 208, 0, 0.4)'
        }}>
          {videometa.label}
        </div> */}
      </div>
      
      {/* 条件渲染 TextCollapseExpand */}
      {videometa.prompt && (
        <TextCollapseExpand 
          title={videometa.title}
          prompt={videometa.prompt}
        />
      )}
    </div>
  );
};

export default VideoCard;