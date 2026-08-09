import { Activity, FolderKanban, GitCompare, Layers3, Network, Zap } from 'lucide-react'

export const NAV_ITEMS = [
  {
    id: 'runs',
    label: 'Runs',
    eyebrow: 'Run overview',
    description: 'See every run side by side and switch between them.',
    shortcut: '1',
    icon: FolderKanban,
  },
  {
    id: 'timeline',
    label: 'Timeline',
    eyebrow: 'Run overview',
    description: 'Follow loss, gradients, and anomaly clusters across the run.',
    shortcut: '2',
    icon: Activity,
  },
  {
    id: 'layers',
    label: 'Layer drill-down',
    eyebrow: 'Layer signals',
    description: 'Trace the layer-level metrics behind an unstable update.',
    shortcut: '3',
    icon: Layers3,
  },
  {
    id: 'moe',
    label: 'Expert utilization',
    eyebrow: 'MoE routing',
    description: 'Per-expert routing shares over time (Mixtral-style models).',
    shortcut: '4',
    icon: Network,
  },
  {
    id: 'diff',
    label: 'Diff view',
    eyebrow: 'State comparison',
    description: 'Compare weight distributions between any two training steps.',
    shortcut: '5',
    icon: GitCompare,
  },
  {
    id: 'spikes',
    label: 'Spike inspector',
    eyebrow: 'Anomaly response',
    description: 'Turn a loss spike into a chronological root-cause story.',
    shortcut: '6',
    icon: Zap,
  },
]
