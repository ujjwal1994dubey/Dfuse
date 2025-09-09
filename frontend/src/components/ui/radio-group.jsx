import * as React from "react"
import { cn } from "../../lib/utils"

const RadioGroup = React.forwardRef(({ 
  className, 
  options = [], 
  value, 
  onChange, 
  name,
  ...props 
}, ref) => {
  const handleChange = (optionValue, event) => {
    // Prevent default radio button behavior
    event.preventDefault()
    
    // If clicking the same value, deselect it
    const newValue = value === optionValue ? "" : optionValue
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
              <div
                key={option}
                className={`flex items-center space-x-2 cursor-pointer hover:bg-accent hover:text-accent-foreground rounded px-2 py-1 transition-colors ${
                  value === option ? 'bg-accent/50' : ''
                }`}
                onClick={(e) => handleChange(option, e)}
              >
                <input
                  type="radio"
                  name={name}
                  value={option}
                  checked={value === option}
                  onChange={() => {}} // Controlled by onClick
                  className="rounded-full border border-input pointer-events-none"
                />
                <span className="text-sm flex-1">{option}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
})
RadioGroup.displayName = "RadioGroup"

export { RadioGroup }
