import { Card, CardContent, CardHeader, CardTitle } from './Card.jsx'
import { cn } from '../../utils.js'

export function ChartCard({ title, description, action, className, children }) {
  return (
    <Card className={cn('chart-card', className)}>
      {(title || description || action) && (
        <CardHeader className="chart-card__header">
          <div className="min-w-0">
            {title && <CardTitle className="chart-card__title">{title}</CardTitle>}
            {description && <p className="chart-card__description">{description}</p>}
          </div>
          {action}
        </CardHeader>
      )}
      <CardContent className="chart-card__content">{children}</CardContent>
    </Card>
  )
}
