import { Card, CardContent, CardHeader, CardTitle } from './Card.jsx'
import { Badge } from './Badge.jsx'
import { cn } from '../../utils.js'

export function StatCard({ label, value, subtitle, icon: Icon, delta, className }) {
  const deltaVariant =
    delta == null ? 'default' : delta > 0 ? 'success' : delta < 0 ? 'danger' : 'default'
  const formattedDelta =
    delta == null
      ? null
      : `${delta > 0 ? '+' : ''}${typeof delta === 'number' ? delta.toFixed(2) : delta}`

  return (
    <Card className={cn('flex flex-col', className)}>
      <CardHeader className="mb-1">
        <CardTitle className="flex items-center gap-2 text-xs font-medium text-muted">
          {Icon && <Icon className="h-4 w-4" />}
          {label}
        </CardTitle>
        {formattedDelta && <Badge variant={deltaVariant}>{formattedDelta}</Badge>}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold tracking-tight text-foreground">{value}</div>
        {subtitle && <div className="mt-1 text-xs text-muted">{subtitle}</div>}
      </CardContent>
    </Card>
  )
}
