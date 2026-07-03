import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { LiveIndicator } from '../LiveIndicator.jsx'

describe('LiveIndicator', () => {
  it('shows Live when connected', () => {
    render(<LiveIndicator status="connected" />)
    expect(screen.getByText('Live')).toBeInTheDocument()
  })

  it('shows Offline when unavailable', () => {
    render(<LiveIndicator status="unavailable" />)
    expect(screen.getByText('Offline')).toBeInTheDocument()
  })
})
