#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""macOS local app UI for Hot Cue Auto Maker."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    import customtkinter as ctk
except ModuleNotFoundError as exc:  # pragma: no cover - dependency error path
    raise SystemExit(
        "customtkinter が見つかりません。`pip install customtkinter` を実行してください。"
    ) from exc

from hotcue_core import ShiftConfig, process_xml, validate_config


APP_NAME = "Hot Cue Auto Maker"
CONFIG_PATH = Path.home() / "Library/Application Support/HotcueAutoMaker/settings.json"


class HotcueApp(ctk.CTk):
    """Desktop UI."""

    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")

        self.title(APP_NAME)
        self.geometry("980x700")
        self.minsize(900, 640)

        saved = self._load_settings()
        defaults = ShiftConfig()

        self.input_path_var = tk.StringVar(value=saved.get("input_path", ""))
        self.output_path_var = tk.StringVar(value=saved.get("output_path", ""))
        self.bpm_threshold_var = tk.StringVar(
            value=str(saved.get("bpm_threshold", defaults.bpm_threshold))
        )
        self.offset_high_var = tk.StringVar(
            value=str(saved.get("offset_high_bpm", defaults.offset_high_bpm))
        )
        self.offset_normal_var = tk.StringVar(
            value=str(saved.get("offset_normal_bpm", defaults.offset_normal_bpm))
        )
        self.energy_min_var = tk.StringVar(
            value=str(saved.get("energy_min", defaults.target_energy_min))
        )
        self.energy_max_var = tk.StringVar(
            value=str(saved.get("energy_max", defaults.target_energy_max))
        )

        self.is_running = False
        self.run_button: ctk.CTkButton | None = None
        self.progress: ctk.CTkProgressBar | None = None
        self.log_box: ctk.CTkTextbox | None = None
        self.status_label: ctk.CTkLabel | None = None

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        header = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 8))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text=APP_NAME,
            font=ctk.CTkFont(family="Avenir Next", size=34, weight="bold"),
            text_color="#0f172a",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text="rekordbox XML のHot Cueを前倒しして、DJミックスの助走点を作成",
            font=ctk.CTkFont(family="Avenir Next", size=15),
            text_color="#334155",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        file_frame = ctk.CTkFrame(self, corner_radius=16, fg_color="#f8fafc")
        file_frame.grid(row=1, column=0, sticky="ew", padx=24, pady=8)
        file_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            file_frame, text="ファイル", font=ctk.CTkFont(family="Avenir Next", size=20, weight="bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=18, pady=(14, 8))

        ctk.CTkLabel(file_frame, text="入力XML").grid(row=1, column=0, sticky="w", padx=(18, 8), pady=8)
        ctk.CTkEntry(file_frame, textvariable=self.input_path_var, height=36).grid(
            row=1, column=1, sticky="ew", padx=8, pady=8
        )
        ctk.CTkButton(file_frame, text="参照", width=88, command=self._select_input).grid(
            row=1, column=2, padx=(8, 18), pady=8
        )

        ctk.CTkLabel(file_frame, text="出力XML").grid(row=2, column=0, sticky="w", padx=(18, 8), pady=(0, 14))
        ctk.CTkEntry(file_frame, textvariable=self.output_path_var, height=36).grid(
            row=2, column=1, sticky="ew", padx=8, pady=(0, 14)
        )
        ctk.CTkButton(file_frame, text="参照", width=88, command=self._select_output).grid(
            row=2, column=2, padx=(8, 18), pady=(0, 14)
        )

        settings_frame = ctk.CTkFrame(self, corner_radius=16, fg_color="#f8fafc")
        settings_frame.grid(row=2, column=0, sticky="ew", padx=24, pady=8)
        settings_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        ctk.CTkLabel(
            settings_frame,
            text="処理設定（可変）",
            font=ctk.CTkFont(family="Avenir Next", size=20, weight="bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=18, pady=(14, 8))

        self._add_setting_input(
            parent=settings_frame,
            row=1,
            col=0,
            label="BPM閾値",
            var=self.bpm_threshold_var,
            help_text="この値を超えると高BPM扱い",
        )
        self._add_setting_input(
            parent=settings_frame,
            row=1,
            col=2,
            label="高BPMオフセット（拍）",
            var=self.offset_high_var,
            help_text="例: 32拍（8小節）",
        )
        self._add_setting_input(
            parent=settings_frame,
            row=2,
            col=0,
            label="通常BPMオフセット（拍）",
            var=self.offset_normal_var,
            help_text="例: 16拍（4小節）",
        )
        self._add_setting_input(
            parent=settings_frame,
            row=2,
            col=2,
            label="対象Energy範囲",
            var=self.energy_min_var,
            second_var=self.energy_max_var,
            help_text="最小〜最大を指定",
        )

        actions = ctk.CTkFrame(self, corner_radius=16, fg_color="#f8fafc")
        actions.grid(row=3, column=0, sticky="nsew", padx=24, pady=(8, 24))
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_rowconfigure(2, weight=1)

        button_row = ctk.CTkFrame(actions, fg_color="transparent")
        button_row.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 8))
        button_row.grid_columnconfigure(3, weight=1)

        self.run_button = ctk.CTkButton(
            button_row,
            text="実行",
            height=40,
            width=160,
            font=ctk.CTkFont(family="Avenir Next", size=16, weight="bold"),
            command=self._run_process,
        )
        self.run_button.grid(row=0, column=0, padx=(0, 8))

        ctk.CTkButton(
            button_row,
            text="設定をデフォルトに戻す",
            height=40,
            width=180,
            fg_color="#e2e8f0",
            text_color="#0f172a",
            hover_color="#cbd5e1",
            command=self._reset_defaults,
        ).grid(row=0, column=1, padx=8)

        self.status_label = ctk.CTkLabel(
            button_row,
            text="準備完了",
            text_color="#1e293b",
            font=ctk.CTkFont(family="Avenir Next", size=14),
        )
        self.status_label.grid(row=0, column=2, padx=(12, 0))

        self.progress = ctk.CTkProgressBar(actions, mode="indeterminate")
        self.progress.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 10))
        self.progress.stop()
        self.progress.set(0)

        self.log_box = ctk.CTkTextbox(actions, corner_radius=12, font=ctk.CTkFont(family="Menlo", size=13))
        self.log_box.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 18))
        self._set_log_text("ログ:\n実行するとここに結果を表示します。")

    def _add_setting_input(
        self,
        parent: ctk.CTkFrame,
        row: int,
        col: int,
        label: str,
        var: tk.StringVar,
        help_text: str,
        second_var: tk.StringVar | None = None,
    ) -> None:
        ctk.CTkLabel(parent, text=label).grid(row=row, column=col, sticky="w", padx=(18, 8), pady=(0, 6))
        if second_var is None:
            ctk.CTkEntry(parent, textvariable=var, height=34).grid(
                row=row, column=col + 1, sticky="ew", padx=(0, 18), pady=(0, 6)
            )
        else:
            row_frame = ctk.CTkFrame(parent, fg_color="transparent")
            row_frame.grid(row=row, column=col + 1, sticky="ew", padx=(0, 18), pady=(0, 6))
            row_frame.grid_columnconfigure((0, 2), weight=1)
            ctk.CTkEntry(row_frame, textvariable=var, height=34).grid(row=0, column=0, sticky="ew")
            ctk.CTkLabel(row_frame, text="〜").grid(row=0, column=1, padx=8)
            ctk.CTkEntry(row_frame, textvariable=second_var, height=34).grid(row=0, column=2, sticky="ew")

        ctk.CTkLabel(
            parent,
            text=help_text,
            text_color="#64748b",
            font=ctk.CTkFont(family="Avenir Next", size=12),
        ).grid(row=row + 10, column=col, columnspan=2, sticky="w", padx=(18, 18), pady=(0, 10))

    def _select_input(self) -> None:
        current = self.input_path_var.get().strip()
        initial_dir = str(Path(current).parent) if current else str(Path.home())
        path = filedialog.askopenfilename(
            title="入力XMLを選択",
            initialdir=initial_dir,
            filetypes=[("XML", "*.xml"), ("All files", "*.*")],
        )
        if not path:
            return
        self.input_path_var.set(path)
        if not self.output_path_var.get().strip():
            self.output_path_var.set(self._suggest_output_path(path))

    def _select_output(self) -> None:
        current = self.output_path_var.get().strip()
        input_path = self.input_path_var.get().strip()
        suggestion = self._suggest_output_path(input_path) if input_path else current
        base = suggestion if suggestion else str(Path.home() / "rekordbox_shifted.xml")
        path = filedialog.asksaveasfilename(
            title="出力XMLを保存",
            initialfile=Path(base).name,
            initialdir=str(Path(base).parent),
            defaultextension=".xml",
            filetypes=[("XML", "*.xml"), ("All files", "*.*")],
        )
        if path:
            self.output_path_var.set(path)

    def _suggest_output_path(self, input_path: str) -> str:
        path = Path(input_path).expanduser()
        suffix = path.suffix if path.suffix else ".xml"
        return str(path.with_name(f"{path.stem}_shifted{suffix}"))

    def _build_config(self) -> ShiftConfig:
        config = ShiftConfig(
            bpm_threshold=float(self.bpm_threshold_var.get().strip()),
            offset_high_bpm=int(self.offset_high_var.get().strip()),
            offset_normal_bpm=int(self.offset_normal_var.get().strip()),
            target_energy_min=int(self.energy_min_var.get().strip()),
            target_energy_max=int(self.energy_max_var.get().strip()),
        )
        validate_config(config)
        return config

    def _run_process(self) -> None:
        if self.is_running:
            return

        try:
            input_raw = self.input_path_var.get().strip()
            output_raw = self.output_path_var.get().strip()
            if not input_raw:
                raise ValueError("入力XMLを指定してください。")
            if not output_raw:
                raise ValueError("出力XMLを指定してください。")
            input_path = Path(input_raw).expanduser()
            output_path = Path(output_raw).expanduser()
            if not input_path.exists():
                raise ValueError("入力XMLが見つかりません。")
            if input_path.resolve() == output_path.resolve():
                raise ValueError("入力XMLと出力XMLは別ファイルにしてください。")
            config = self._build_config()
        except ValueError as exc:
            messagebox.showerror("入力エラー", str(exc))
            return

        self._save_settings()
        self.is_running = True
        if self.run_button:
            self.run_button.configure(state="disabled")
        if self.status_label:
            self.status_label.configure(text="処理中...")
        if self.progress:
            self.progress.start()
        self._set_log_text(
            "\n".join(
                [
                    "処理開始",
                    f"入力: {input_path}",
                    f"出力: {output_path}",
                    (
                        "設定: BPM>{:.2f} → -{}拍 / それ以外 → -{}拍 / Energy {}-{}".format(
                            config.bpm_threshold,
                            config.offset_high_bpm,
                            config.offset_normal_bpm,
                            config.target_energy_min,
                            config.target_energy_max,
                        )
                    ),
                    "",
                ]
            )
        )

        worker = threading.Thread(
            target=self._process_worker, args=(str(input_path), str(output_path), config), daemon=True
        )
        worker.start()

    def _process_worker(self, input_path: str, output_path: str, config: ShiftConfig) -> None:
        try:
            stats = process_xml(input_path, output_path, config)
        except Exception as exc:  # pragma: no cover - UI error path
            self.after(0, self._on_process_error, str(exc))
            return
        self.after(0, self._on_process_success, output_path, stats)

    def _on_process_success(self, output_path: str, stats: dict) -> None:
        self.is_running = False
        if self.run_button:
            self.run_button.configure(state="normal")
        if self.status_label:
            self.status_label.configure(text="完了")
        if self.progress:
            self.progress.stop()
            self.progress.set(0)
        self._append_log("処理完了")
        self._append_log(f"  処理トラック数: {stats['tracks_processed']}")
        self._append_log(f"  シフトしたHot Cue: {stats['shifted']}")
        self._append_log(f"  スキップ（マイナス）: {stats['skipped_negative']}")
        self._append_log(f"  スキップ（対象外Energy）: {stats['skipped_not_target']}")
        self._append_log(f"出力: {output_path}")
        messagebox.showinfo("完了", "処理が完了しました。")

    def _on_process_error(self, error_text: str) -> None:
        self.is_running = False
        if self.run_button:
            self.run_button.configure(state="normal")
        if self.status_label:
            self.status_label.configure(text="エラー")
        if self.progress:
            self.progress.stop()
            self.progress.set(0)
        self._append_log(f"エラー: {error_text}")
        messagebox.showerror("処理エラー", error_text)

    def _reset_defaults(self) -> None:
        defaults = ShiftConfig()
        self.bpm_threshold_var.set(str(defaults.bpm_threshold))
        self.offset_high_var.set(str(defaults.offset_high_bpm))
        self.offset_normal_var.set(str(defaults.offset_normal_bpm))
        self.energy_min_var.set(str(defaults.target_energy_min))
        self.energy_max_var.set(str(defaults.target_energy_max))

    def _set_log_text(self, text: str) -> None:
        if not self.log_box:
            return
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.insert("1.0", text)
        self.log_box.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        if not self.log_box:
            return
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{text}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _load_settings(self) -> dict:
        if not CONFIG_PATH.exists():
            return {}
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self) -> None:
        payload = {
            "input_path": self.input_path_var.get().strip(),
            "output_path": self.output_path_var.get().strip(),
            "bpm_threshold": self.bpm_threshold_var.get().strip(),
            "offset_high_bpm": self.offset_high_var.get().strip(),
            "offset_normal_bpm": self.offset_normal_var.get().strip(),
            "energy_min": self.energy_min_var.get().strip(),
            "energy_max": self.energy_max_var.get().strip(),
        }
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _on_close(self) -> None:
        self._save_settings()
        self.destroy()


def main() -> None:
    app = HotcueApp()
    app.mainloop()


if __name__ == "__main__":
    main()
