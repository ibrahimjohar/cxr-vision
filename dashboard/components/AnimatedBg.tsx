export default function AnimatedBg() {
  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 0,
        pointerEvents: 'none',
        overflow: 'hidden',
      }}
    >
      <div
        className="orb-1"
        style={{
          position: 'absolute',
          width: '600px',
          height: '600px',
          borderRadius: '50%',
          background: 'rgba(58, 1, 92, 0.5)',
          filter: 'blur(140px)',
          transform: 'translate(-50%, -50%)',
        }}
      />
      <div
        className="orb-2"
        style={{
          position: 'absolute',
          width: '500px',
          height: '500px',
          borderRadius: '50%',
          background: 'rgba(50, 0, 79, 0.45)',
          filter: 'blur(140px)',
          transform: 'translate(-50%, -50%)',
        }}
      />
      <div
        className="orb-3"
        style={{
          position: 'absolute',
          width: '450px',
          height: '450px',
          borderRadius: '50%',
          background: 'rgba(75, 15, 130, 0.35)',
          filter: 'blur(140px)',
          transform: 'translate(-50%, -50%)',
        }}
      />
    </div>
  )
}