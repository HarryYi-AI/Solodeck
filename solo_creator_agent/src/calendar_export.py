from __future__ import annotations

from datetime import datetime, timedelta, timezone
from html import escape


def _ics_escape(value: str) -> str:
    return (
        str(value or "")
        .replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\n", "\\n")
    )


def _priority_value(priority: str) -> int:
    return {"high": 1, "medium": 5, "low": 9}.get(priority, 5)


def task_due_time(index: int, priority: str) -> datetime:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    if priority == "high":
        return now + timedelta(hours=1 + index)
    if priority == "medium":
        return now + timedelta(days=1, hours=index)
    return now + timedelta(days=3 + index)


def _parse_task_due(task: dict, index: int) -> datetime:
    due_at = task.get("due_at") or task.get("deadline")
    if due_at:
        try:
            parsed = datetime.fromisoformat(str(due_at).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            pass
    return task_due_time(index, task.get("priority", "medium"))


def tasks_to_ics(tasks: list[dict], calendar_name: str = "SoloDeck Tasks", limit: int = 12) -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//SoloDeck//Creator Task Calendar//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{_ics_escape(calendar_name)}",
    ]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    active = [task for task in tasks if not task.get("done")][:limit]
    for index, task in enumerate(active):
        due = _parse_task_due(task, index)
        end = due + timedelta(minutes=30)
        uid = task.get("id") or f"solodeck-{index}-{stamp}"
        title = task.get("title", "SoloDeck Task")
        detail = task.get("detail", "")
        priority = task.get("priority", "medium")
        lines.extend([
            "BEGIN:VEVENT",
            f"UID:{_ics_escape(uid)}@solodeck",
            f"DTSTAMP:{stamp}",
            f"DTSTART:{due.strftime('%Y%m%dT%H%M%SZ')}",
            f"DTEND:{end.strftime('%Y%m%dT%H%M%SZ')}",
            f"SUMMARY:{_ics_escape(title)}",
            f"DESCRIPTION:{_ics_escape(detail)}",
            f"PRIORITY:{_priority_value(priority)}",
            "BEGIN:VALARM",
            "TRIGGER:-PT10M",
            "ACTION:DISPLAY",
            f"DESCRIPTION:{_ics_escape(title)}",
            "END:VALARM",
            "END:VEVENT",
        ])
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def browser_notification_widget(tasks: list[dict], language: str = "中文", limit: int = 5) -> str:
    zh = language == "中文"
    active = [task for task in tasks if not task.get("done")][:limit]
    payload = [
        {
            "title": str(task.get("title", "SoloDeck")),
            "body": str(task.get("detail", ""))[:120],
            "delayMinutes": 1 if task.get("priority") == "high" else 3 if task.get("priority") == "medium" else 5,
        }
        for task in active
    ]
    button = "开启提醒" if zh else "Enable Reminders"
    note = "也可以下载日历文件，加入系统日历。" if zh else "You can also download the calendar file and add it to your system calendar."
    empty = "暂无可提醒任务" if zh else "No tasks to remind"
    import json

    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
      <button id="solodeckNotifyBtn" style="border:1px solid #d8d2c8;background:#fffefc;border-radius:999px;padding:8px 14px;cursor:pointer;color:#111827;">{escape(button)}</button>
      <span id="solodeckNotifyStatus" style="margin-left:10px;color:#6b7280;font-size:13px;">{escape(note)}</span>
    </div>
    <script>
    const tasks = {json.dumps(payload, ensure_ascii=False)};
    const statusEl = document.getElementById("solodeckNotifyStatus");
    document.getElementById("solodeckNotifyBtn").onclick = async () => {{
      if (!tasks.length) {{
        statusEl.textContent = "{escape(empty)}";
        return;
      }}
      if (!("Notification" in window)) {{
        statusEl.textContent = "This browser does not support notifications.";
        return;
      }}
      const permission = await Notification.requestPermission();
      if (permission !== "granted") {{
        statusEl.textContent = "Notification permission was not granted.";
        return;
      }}
      tasks.forEach((task) => {{
        window.setTimeout(() => {{
          new Notification(task.title, {{ body: task.body || "SoloDeck task reminder" }});
        }}, Math.max(1, task.delayMinutes) * 60 * 1000);
      }});
      statusEl.textContent = tasks.length + " reminders scheduled.";
    }};
    </script>
    """
