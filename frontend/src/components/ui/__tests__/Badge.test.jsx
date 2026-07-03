import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { Badge } from '../Badge.jsx'

describe('Badge', () => {
  it('renders text', () => {
    render(<Badge>3 spikes</Badge>)
    expect(screen.getByText('3 spikes')).toBeInTheDocument()
  })
})
