param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string]$TestRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes, System.Windows.Forms
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class NativeEditInput {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr window);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint x, uint y, uint data, UIntPtr extra);
    public static void Click(int x, int y) {
        if (!SetCursorPos(x, y)) throw new InvalidOperationException("Cannot position the test pointer");
        mouse_event(2, 0, 0, 0, UIntPtr.Zero);
        mouse_event(4, 0, 0, 0, UIntPtr.Zero);
    }
}
'@

function Wait-Until([scriptblock]$Check, [string]$Description) {
    $deadline = [DateTime]::UtcNow.AddSeconds(15)
    do {
        try { if (& $Check) { return } }
        catch [System.Windows.Automation.ElementNotAvailableException] {
            # A WebView rerender can replace the element between the query and read.
        }
        Start-Sleep -Milliseconds 50
    } while ([DateTime]::UtcNow -lt $deadline)
    $windows = [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
        [System.Windows.Automation.TreeScope]::Children,
        [System.Windows.Automation.Condition]::TrueCondition
    )
    foreach ($visible in $windows) {
        if ($visible.Current.ProcessId -ne $script:process.Id) { continue }
        Write-Output "Window: $($visible.Current.Name), class=$($visible.Current.ClassName)"
        foreach ($element in $visible.FindAll([System.Windows.Automation.TreeScope]::Descendants,
            [System.Windows.Automation.Condition]::TrueCondition)) {
            Write-Output "UI: $($element.Current.ControlType.ProgrammaticName) | $($element.Current.Name) | $($element.Current.AutomationId)"
        }
    }
    throw "Timed out: $Description"
}

function Find-Names([string[]]$Names) {
    $elements = $script:window.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants,
        [System.Windows.Automation.Condition]::TrueCondition
    )
    foreach ($element in $elements) {
        if ($element.Current.Name -in $Names) { return $element }
    }
    return $null
}

function Click-Element($Element) {
    if ($null -eq $Element) { throw 'Required UI element is absent' }
    $bounds = $Element.Current.BoundingRectangle
    if ($bounds.IsEmpty) { throw "UI element is not visible: $($Element.Current.Name)" }
    [NativeEditInput]::Click(
        [int]($bounds.X + $bounds.Width / 2),
        [int]($bounds.Y + $bounds.Height / 2)
    )
}

function Send-Keys([string]$Keys) { [System.Windows.Forms.SendKeys]::SendWait($Keys) }

function Read-Value($Element) {
    $pattern = $null
    if ($Element.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$pattern)) {
        return $pattern.Current.Value
    }
    if ($Element.TryGetCurrentPattern([System.Windows.Automation.TextPattern]::Pattern, [ref]$pattern)) {
        return $pattern.DocumentRange.GetText(-1).TrimEnd("`r", "`n")
    }
    throw "Text control exposes no readable value: $($Element.Current.Name)"
}

function Expect-Value($Element, [string]$Expected, [string]$Description) {
    Wait-Until { (Read-Value $Element) -ceq $Expected } $Description
    Write-Output "PASS: $Description"
}

function Qualify-Text($Element, [string]$Label) {
    Click-Element $Element
    Send-Keys '^a'
    Send-Keys 'first'
    Expect-Value $Element 'first' "$Label select-all and input"
    [System.Windows.Forms.Clipboard]::SetText('second')
    Send-Keys '^a'
    Send-Keys '^v'
    Expect-Value $Element 'second' "$Label paste"
    Send-Keys '^z'
    Expect-Value $Element 'first' "$Label undo"
    Send-Keys '^y'
    Expect-Value $Element 'second' "$Label redo"
    Send-Keys '^a'
    Send-Keys '^c'
    Wait-Until { [System.Windows.Forms.Clipboard]::GetText() -ceq 'second' } "$Label copy"
    Write-Output "PASS: $Label copy"
    Send-Keys '^x'
    Expect-Value $Element '' "$Label cut"
    Send-Keys '^v'
    Expect-Value $Element 'second' "$Label paste cut text"
    Send-Keys '^a'
    Send-Keys '{BACKSPACE}'
    Expect-Value $Element '' "$Label clear"
}

function Clip-Elements {
    $elements = $script:window.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants,
        [System.Windows.Automation.Condition]::TrueCondition
    )
    @($elements | Where-Object { $_.Current.Name -match '^Clip [^ ]+ on ' })
}

function Folder-Dialog {
    $condition = [System.Windows.Automation.AndCondition]::new(
        [System.Windows.Automation.PropertyCondition]::new(
            [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $script:process.Id
        ),
        [System.Windows.Automation.PropertyCondition]::new(
            [System.Windows.Automation.AutomationElement]::ClassNameProperty, '#32770'
        )
    )
    [System.Windows.Automation.AutomationElement]::RootElement.FindFirst(
        [System.Windows.Automation.TreeScope]::Descendants, $condition
    )
}

New-Item -ItemType Directory -Force -Path $TestRoot | Out-Null
$project = Join-Path $TestRoot 'NativeEdit.opentake'
New-Item -ItemType Directory -Force -Path (Join-Path $project 'media') | Out-Null
'{"fps":30,"width":1920,"height":1080,"settingsConfigured":true,"tracks":[]}' |
    Set-Content -Encoding utf8 (Join-Path $project 'project.json')
'{"version":1,"entries":[],"folders":[],"favorites":[]}' |
    Set-Content -Encoding utf8 (Join-Path $project 'media.json')
$process = Start-Process -FilePath $Executable -PassThru
try {
    $script:window = $null
    Wait-Until {
        $condition = [System.Windows.Automation.PropertyCondition]::new(
            [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $process.Id
        )
        $script:window = [System.Windows.Automation.AutomationElement]::RootElement.FindFirst(
            [System.Windows.Automation.TreeScope]::Children, $condition
        )
        $null -ne $script:window
    } 'application window'
    if (-not [NativeEditInput]::SetForegroundWindow($script:window.Current.NativeWindowHandle)) {
        throw 'Cannot activate the qualification application'
    }
    # Each runner starts with a fresh WebView profile. The first-run notice
    # covers the launcher, so dismiss it through its real UI before opening.
    Wait-Until { $null -ne (Find-Names @('Get started', '开始')) } 'first-run welcome notice'
    Click-Element (Find-Names @('Get started', '开始'))
    Wait-Until { $null -eq (Find-Names @('Get started', '开始')) } 'welcome notice dismissed'
    Wait-Until { $null -ne (Find-Names @('Open Project', '打开项目')) } 'frontend readiness'
    Click-Element (Find-Names @('Open Project', '打开项目'))
    $script:dialog = $null
    Wait-Until { $script:dialog = Folder-Dialog; $null -ne $script:dialog } 'native project folder picker'
    if (-not [NativeEditInput]::SetForegroundWindow($script:dialog.Current.NativeWindowHandle)) {
        throw 'Cannot activate the project folder picker'
    }
    # The native folder picker accepts an address through Ctrl+L. This grants
    # exactly the disposable fixture through the same dialog as a real user.
    Send-Keys '^l'
    [System.Windows.Forms.Clipboard]::SetText($project)
    Send-Keys '^v'
    Send-Keys '{ENTER}'
    $script:selectFolder = $null
    Wait-Until {
        $script:selectFolder = $script:dialog.FindFirst(
            [System.Windows.Automation.TreeScope]::Descendants,
            [System.Windows.Automation.AndCondition]::new(
                [System.Windows.Automation.PropertyCondition]::new(
                    [System.Windows.Automation.AutomationElement]::AutomationIdProperty, '1'
                ),
                [System.Windows.Automation.PropertyCondition]::new(
                    [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
                    [System.Windows.Automation.ControlType]::Button
                )
            )
        )
        $null -ne $script:selectFolder -and $script:selectFolder.Current.IsEnabled
    } 'folder selection button'
    Click-Element $script:selectFolder
    Wait-Until { $null -eq (Folder-Dialog) } 'project folder accepted'
    $search = $null
    Wait-Until { $script:search = Find-Names @('Search', '搜索'); $null -ne $script:search } 'editor search input'
    Qualify-Text $script:search 'input'
    $chat = Find-Names @('Chat', '对话')
    Click-Element $chat
    $textarea = $null
    Wait-Until {
        $edits = $script:window.FindAll(
            [System.Windows.Automation.TreeScope]::Descendants,
            [System.Windows.Automation.PropertyCondition]::new(
                [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
                [System.Windows.Automation.ControlType]::Edit
            )
        )
        $script:textarea = @($edits | Where-Object { $_.Current.Name -notin @('Search', '搜索') }) | Select-Object -First 1
        $null -ne $script:textarea
    } 'agent textarea'
    Qualify-Text $script:textarea 'textarea'
    Click-Element (Find-Names @('Add Text', '添加文本'))
    Wait-Until { @(Clip-Elements).Count -eq 1 } 'one added timeline clip'
    Send-Keys '^c'
    Click-Element (Find-Names @('Jump to End', '跳到结尾'))
    Send-Keys '^v'
    Wait-Until { @(Clip-Elements).Count -eq 2 } 'timeline copy and paste'
    Write-Output 'PASS: timeline copy and paste'
    Click-Element (@(Clip-Elements)[1])
    Send-Keys '^x'
    Wait-Until { @(Clip-Elements).Count -eq 1 } 'timeline cut'
    Write-Output 'PASS: timeline cut'
    Send-Keys '^v'
    Wait-Until { @(Clip-Elements).Count -eq 2 } 'timeline paste cut clip'
    Send-Keys '{BACKSPACE}'
    Wait-Until { @(Clip-Elements).Count -eq 1 } 'timeline delete'
    Write-Output 'PASS: timeline delete'
    Send-Keys '^z'
    Wait-Until { @(Clip-Elements).Count -eq 2 } 'timeline undo'
    Write-Output 'PASS: timeline undo'
} finally {
    if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
}
