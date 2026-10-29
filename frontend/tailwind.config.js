/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'media',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        page: 'var(--page)',
        surface: 'var(--surface)',
        border: 'var(--border)',
        'border-strong': 'var(--border-strong)',
        ink: 'var(--ink)',
        muted: 'var(--muted)',
        subtle: 'var(--subtle)',
        red: 'var(--red)',
        blue: 'var(--blue)',
        amber: 'var(--amber)',
        hover: 'var(--hover)',
      },
      borderRadius: {
        panel: 'var(--radius-panel)',
        control: 'var(--radius-control)',
        lg: 'var(--radius-panel)',
        md: 'var(--radius-control)',
        sm: '6px',
      },
      fontFamily: {
        sans: ['"Geist Variable"', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
