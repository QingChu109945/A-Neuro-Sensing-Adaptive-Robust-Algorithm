import time
import sys
from datetime import timedelta
from typing import Optional

# 非 ASCII 装饰字符 -> ASCII 替代 (GBK 等非 Unicode 代码页下的降级映射)
_ASCII_FALLBACK = (
    ("█", "#"), ("▓", "#"), ("▒", "+"), ("░", "-"),
    ("✓", "v"), ("✔", "v"), ("✗", "x"), ("○", "o"),
    ("●", "*"), ("⏸", "||"),
)


def _out(text: str):
    """容错写 stdout: 终端为 GBK 等代码页 (如重定向/管道) 时,
    █░✓○ 等字符无法编码会抛 UnicodeEncodeError 中断实验;
    此处失败后降级为 ASCII 替代字符重写, 保证任何终端下都能输出。"""
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        for ch, rep in _ASCII_FALLBACK:
            text = text.replace(ch, rep)
        try:
            sys.stdout.write(text)
        except UnicodeEncodeError:
            enc = sys.stdout.encoding or "ascii"
            sys.stdout.write(text.encode(enc, "replace").decode(enc))
    sys.stdout.flush()


class ProgressBar:
    """命令行进度条显示工具"""
    
    def __init__(self, total: int = 100, description: str = "", 
                 bar_length: int = 40, show_eta: bool = True):
        self.total = total
        self.description = description
        self.bar_length = bar_length
        self.show_eta = show_eta
        self._current = 0
        self._start_time = time.time()
        self._last_update_time = 0
        self._update_interval = 0.1
    
    def update(self, current: int, message: str = ""):
        """更新进度"""
        self._current = current
        now = time.time()
        
        if now - self._last_update_time < self._update_interval and current != self.total:
            return
        
        self._last_update_time = now
        self._display(message)
    
    def increment(self, step: int = 1, message: str = ""):
        """增加进度"""
        self._current = min(self._current + step, self.total)
        self.update(self._current, message)
    
    def finish(self, message: str = ""):
        """完成进度"""
        self._current = self.total
        self._display(message, force=True)
        print()
    
    def _display(self, message: str = "", force: bool = False):
        """显示进度条"""
        progress = min(self._current / self.total, 1.0)
        percent = int(progress * 100)
        
        filled_length = int(self.bar_length * progress)
        bar = "█" * filled_length + "░" * (self.bar_length - filled_length)
        
        elapsed_time = time.time() - self._start_time
        eta = ""
        if self.show_eta and progress > 0:
            estimated_total = elapsed_time / progress
            remaining = estimated_total - elapsed_time
            eta = f" ETA: {timedelta(seconds=int(remaining))}"
        
        elapsed_str = f" [{timedelta(seconds=int(elapsed_time))}]"
        
        line = f"\r{self.description} [{bar}] {percent}%{elapsed_str}{eta}"
        if message:
            line += f" - {message}"

        _out(line)

class ProgressTracker:
    """多任务进度跟踪器"""
    
    def __init__(self):
        self._tasks = {}
        self._current_task = None
    
    def add_task(self, task_name: str, total: int, description: str = "") -> ProgressBar:
        """添加任务"""
        progress = ProgressBar(total=total, description=description)
        self._tasks[task_name] = progress
        return progress
    
    def start_task(self, task_name: str):
        """开始任务"""
        if task_name in self._tasks:
            self._current_task = task_name
            print(f"\n{'='*60}")
            print(f"Starting: {task_name}")
            print(f"{'='*60}")
    
    def end_task(self, task_name: str):
        """结束任务"""
        if task_name in self._tasks:
            self._tasks[task_name].finish()
            if self._current_task == task_name:
                self._current_task = None
    
    def get_task(self, task_name: str) -> Optional[ProgressBar]:
        """获取任务进度条"""
        return self._tasks.get(task_name)

class StatusIndicator:
    """状态指示器 - 显示程序运行状态"""
    
    def __init__(self):
        self._spinner_chars = ['|', '/', '-', '\\']
        self._spinner_index = 0
        self._last_update_time = 0
    
    def show_busy(self, message: str = "Processing..."):
        """显示忙碌状态"""
        now = time.time()
        if now - self._last_update_time < 0.1:
            return
        
        self._last_update_time = now
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_chars)
        
        line = f"\r{self._spinner_chars[self._spinner_index]} {message}"
        _out(line)
    
    def show_idle(self, message: str = "Waiting..."):
        """显示空闲状态"""
        line = f"\r○ {message}"
        _out(line)
    
    def show_completed(self, message: str = "Completed"):
        """显示完成状态"""
        line = f"\r✓ {message}\n"
        _out(line)
    
    def clear(self):
        """清除状态显示"""
        sys.stdout.write("\r")
        sys.stdout.flush()

def format_duration(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"

def log_progress(message: str, progress: int, total: int):
    """简单的进度日志"""
    percent = int(progress / total * 100)
    print(f"[{percent:3d}%] {message}")

def log_status(message: str, status: str = "INFO"):
    """状态日志"""
    timestamp = time.strftime("%H:%M:%S")
    status_colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "DEBUG": "\033[96m"
    }
    color = status_colors.get(status, "")
    print(f"[{timestamp}] {color}{status}\033[0m: {message}")