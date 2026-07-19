from pathlib import Path

d = Path.home() / ".streamlit"
d.mkdir(exist_ok=True)
(d / "credentials.toml").write_text('[general]\nemail = ""\n', encoding="utf-8")
(d / "config.toml").write_text(
    "\n".join(
        [
            "[browser]",
            "gatherUsageStats = false",
            "",
            "[server]",
            "headless = true",
            'address = "127.0.0.1"',
            "port = 8501",
            "",
        ]
    ),
    encoding="utf-8",
)
print("ok", d)
print((d / "config.toml").read_text(encoding="utf-8"))
