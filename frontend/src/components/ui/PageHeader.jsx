import { cn } from '../../utils.js'

export function PageHeader({ eyebrow, title, description, icon: Icon, className, children }) {
  return (
    <section className={cn('page-header', className)}>
      <div className="flex min-w-0 items-start gap-3 sm:gap-4">
        {Icon && (
          <div className="page-header__icon" aria-hidden="true">
            <Icon className="h-5 w-5" />
          </div>
        )}
        <div className="min-w-0">
          <p className="page-header__eyebrow">{eyebrow}</p>
          <h1 className="page-header__title">{title}</h1>
          <p className="page-header__description">{description}</p>
        </div>
      </div>
      {children && <div className="page-header__actions">{children}</div>}
    </section>
  )
}
