import { Component } from 'react'
import { Button } from './ui/Button.jsx'

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
        <div className="rounded-lg border border-border bg-panel p-10 text-center">
          <h2 className="mb-2 text-lg font-semibold text-danger">Something went wrong.</h2>
          <p className="mb-4 text-sm text-muted">
            {this.state.error?.message || 'An unexpected error occurred.'}
          </p>
          <Button variant="primary" onClick={this.handleReset}>
            Try again
          </Button>
        </div>
      )
    }

    return this.props.children
  }
}
