import { Activity, GitCompare, Layers3, Zap } from 'lucide-react'

export const NAV_ITEMS = [
  {
    id: 'timeline',
    label: 'Timeline',
    eyebrow: 'Run overview',
    description: 'Follow loss, gradients, and anomaly clusters across the run.',
    shortcut: '1',
    icon: Activity,
  },
  {
    id: 'layers',
    label: 'Layer drill-down',
    eyebrow: 'Layer signals',
    description: 'Trace the layer-level metrics behind an unstable update.',
    shortcut: '2',
    icon: Layers3,
  },
  {
    id: 'diff',
    label: 'Diff view',
    eyebrow: 'State comparison',
    description: 'Compare weight distributions between any two training steps.',
    shortcut: '3',
    icon: GitCompare,
  },
  {
    id: 'spikes',
    label: 'Spike inspector',
    eyebrow: 'Anomaly response',
    description: 'Turn a loss spike into a chronological root-cause story.',
    shortcut: '4',
    icon: Zap,
  },
]
