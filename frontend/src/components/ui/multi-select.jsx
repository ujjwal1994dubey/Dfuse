import * as React from "react"
import { cn } from "../../lib/utils"
import { Badge } from "./badge"

const MultiSelect = React.forwardRef(({ 
  className, 
  options = [], 
  value = [], 
  onChange, 
  placeholder = "Select items...",
  ...props 
}, ref) => {
  const handleToggle = (optionValue) => {
    const newValue = value.includes(optionValue)
      ? value.filter(v => v !== optionValue)
      : [...value, optionValue]
    onChange?.(newValue)
  }

  return (
    <div className={cn("space-y-2", className)} ref={ref} {...props}>
      <div className="max-h-32 overflow-y-auto border rounded-md p-2 bg-background">
        {options.length === 0 ? (
          <p className="text-sm text-muted-foreground py-2 text-center">
            No options available
          </p>
        ) : (
          <div className="space-y-1">
            {options.map((option) => (
              <label
                key={option}
                className="flex items-center space-x-2 cursor-pointer hover:bg-accent hover:text-accent-foreground rounded px-2 py-1 transition-colors"
              >
                <input
                  type="checkbox"
                  checked={value.includes(option)}
                  onChange={() => handleToggle(option)}
                  className="rounded border border-input"
                />
                <span className="text-sm">{option}</span>
              </label>
            ))}
          </div>
        )}
      </div>
      
      {value.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {value.map((selected) => (
            <Badge
              key={selected}
              variant="secondary"
              className="text-xs cursor-pointer hover:bg-destructive hover:text-destructive-foreground"
              onClick={() => handleToggle(selected)}
            >
              {selected} ×
            </Badge>
          ))}
        </div>
      )}
    </div>
  )
})
MultiSelect.displayName = "MultiSelect"

export { MultiSelect }
