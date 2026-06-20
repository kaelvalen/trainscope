import { Component } from 'react'
import { COLORS } from '../theme.js'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    if (this.props.onError) {
      this.props.onError(error, errorInfo)
    } else {
      // eslint-disable-next-line no-console
      console.error('ErrorBoundary caught an error:', error, errorInfo)
    }
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null })
    if (this.props.onReset) {
      this.props.onReset()
    }
  }

  render() {
    if (this.state.hasError) {
      const custom = this.props.fallback
      if (typeof custom === 'function') {
        return custom(this.state.error, this.handleReset)
      }
      if (custom) return custom

      return (
        <div
          style={{
            padding: '40px',
            textAlign: 'center',
            color: COLORS.danger,
            background: COLORS.bg,
            borderRadius: '6px',
            border: `1px solid ${COLORS.border}`,
          }}
        >
          <h2 style={{ fontSize: '18px', marginBottom: '12px' }}>Something went wrong.</h2>
          <p style={{ color: COLORS.muted, fontSize: '13px', marginBottom: '16px' }}>
            {this.state.error?.message || 'An unexpected error occurred.'}
          </p>
          <button
            onClick={this.handleReset}
            style={{
              background: COLORS.button,
              color: '#fff',
              border: 'none',
              borderRadius: '6px',
              padding: '8px 16px',
              cursor: 'pointer',
              fontSize: '13px',
            }}
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
