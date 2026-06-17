'use client'

import { useState } from 'react'

export default function HomePage() {
  const [count, setCount] = useState(0)

  return (
    <div style={{ padding: '4rem', color: 'white' }}>
      <h1 style={{ fontSize: '3rem', marginBottom: '2rem' }}>hello world</h1>
      <button
        onClick={() => setCount(c => c + 1)}
        style={{
          padding: '1rem 2rem',
          background: 'purple',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          fontSize: '1rem',
          cursor: 'pointer',
        }}
      >
        clicked {count} times
      </button>
    </div>
  )
}