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
    <Card className={cn('metric-card flex flex-col', className)}>
      <CardHeader className="metric-card__header mb-2">
        <CardTitle className="metric-card__label flex items-center gap-2 text-xs font-medium text-muted">
          {Icon && (
            <span className="metric-card__icon" aria-hidden="true">
              <Icon className="h-3.5 w-3.5" />
            </span>
          )}
          <span>{label}</span>
        </CardTitle>
        {formattedDelta && <Badge variant={deltaVariant}>{formattedDelta}</Badge>}
      </CardHeader>
      <CardContent className="metric-card__content">
        <div className="metric-card__value">{value}</div>
        {subtitle && <div className="metric-card__subtitle">{subtitle}</div>}
      </CardContent>
    </Card>
  )
}
