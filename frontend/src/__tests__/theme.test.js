import { afterEach, expect, test } from 'vitest'
import { CHART_COLORS, readToken } from '../theme.js'

afterEach(() => document.documentElement.style.removeProperty('--ink'))

test('readToken resolves a CSS variable to its value', () => {
  document.documentElement.style.setProperty('--ink', '#171717')
  expect(readToken('--ink')).toBe('#171717')
})

test('chart colors are bound to theme tokens', () => {
  expect(CHART_COLORS.spike).toBe('var(--red)')
  expect(CHART_COLORS.loss).toBe('var(--ink)')
  expect(CHART_COLORS.gradNorm).toBe('var(--blue)')
})
