/**
 * Join conditional class names into a single string.
 *
 * Mirrors the common `clsx` / `tailwind-merge` pattern without adding extra
 * dependencies for this project.
 */
export function cn(...inputs) {
  return inputs.filter(Boolean).join(' ')
}
