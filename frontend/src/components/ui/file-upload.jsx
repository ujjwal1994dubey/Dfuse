import * as React from "react"
import { cn } from "../../lib/utils"
import { Button } from "./button"
import { Upload } from "lucide-react"

const FileUpload = React.forwardRef(({ className, onFileChange, accept, children, ...props }, ref) => {
  const inputRef = React.useRef(null)

  const handleClick = () => {
    inputRef.current?.click()
  }

  const handleChange = (event) => {
    const file = event.target.files?.[0]
    if (file && onFileChange) {
      onFileChange(file)
    }
  }

  return (
    <div className={cn("", className)} {...props}>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={handleChange}
        className="hidden"
        {...props}
      />
      <Button
        type="button"
        variant="outline"
        onClick={handleClick}
        className="w-full justify-center gap-2"
      >
        <Upload size={16} />
        {children || "Choose File"}
      </Button>
    </div>
  )
})
FileUpload.displayName = "FileUpload"

export { FileUpload }
