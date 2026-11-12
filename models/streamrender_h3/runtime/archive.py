"""Persist interactive sessions and build comparison videos."""
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw


class SessionArchive:
    def __init__(self, settings, root):
        self.root = Path(settings.get("output_dir", str(root / "outputs")))
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="session-archive"
        )

    def session_dir(self, session):
        return self.root / session.id

    def start(self, session):
        self.session_dir(session).mkdir(parents=True, exist_ok=True)
        session.archive_futures = []
        session.archive_rendered_frames = 0

    @staticmethod
    def _write_semantic(target, records):
        target.mkdir(parents=True, exist_ok=True)
        for frame_index, png in records:
            (target / f"{frame_index:06d}.png").write_bytes(png)

    def save_semantic(self, session, records):
        payload = [
            (int(frame_index), bytes(png))
            for frame_index, png, _, _ in records
        ]
        future = self.executor.submit(
            self._write_semantic,
            self.session_dir(session) / "semantic",
            payload,
        )
        session.archive_futures.append(future)

    def save_rendered(self, session, images):
        target = self.session_dir(session) / "rendered"
        target.mkdir(parents=True, exist_ok=True)
        first = int(session.archive_rendered_frames)
        for offset, image in enumerate(images):
            (target / f"{first + offset:06d}.jpg").write_bytes(image)
        session.archive_rendered_frames = first + len(images)

    @staticmethod
    def _video_codec(av):
        for name in ("libx264", "h264", "mpeg4"):
            try:
                av.codec.Codec(name, "w")
            except Exception:
                continue
            return name
        raise RuntimeError("No MP4 video encoder is available in PyAV")

    def _finalize(self, session, writes):
        output_dir = self.session_dir(session)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "archive.json"
        temporary = output_dir / "comparison.part.mp4"
        report = {
            "session": session.id,
            "fps": 24,
            "status": "building",
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n")

        try:
            for future in writes:
                future.result()

            semantic = sorted((output_dir / "semantic").glob("*.png"))
            rendered = sorted((output_dir / "rendered").glob("*.jpg"))
            count = min(len(semantic), len(rendered))

            report.update(
                semantic_frames=len(semantic),
                rendered_frames=len(rendered),
                paired_frames=count,
            )

            if count == 0:
                report["status"] = "no_paired_frames"
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                logging.warning(
                    "H3_ARCHIVE_EMPTY session=%s", session.id
                )
                return

            import av

            with Image.open(rendered[0]) as first:
                width, height = first.size

            codec = self._video_codec(av)
            resampling = getattr(Image, "Resampling", Image)
            container = av.open(str(temporary), mode="w")

            try:
                stream = container.add_stream(codec, rate=24)
                stream.width = width * 2
                stream.height = height
                stream.pix_fmt = "yuv420p"

                if codec == "libx264":
                    stream.options = {
                        "crf": "18",
                        "preset": "medium",
                    }

                pairs = zip(
                    semantic[:count],
                    rendered[:count],
                    strict=True,
                )

                for input_path, output_path in pairs:
                    with Image.open(input_path) as source:
                        left = source.convert("RGB").resize(
                            (width, height),
                            resampling.NEAREST,
                        )

                    with Image.open(output_path) as generated:
                        right = generated.convert("RGB")
                        if right.size != (width, height):
                            right = right.resize(
                                (width, height),
                                resampling.LANCZOS,
                            )

                    comparison = Image.new(
                        "RGB", (width * 2, height)
                    )
                    comparison.paste(left, (0, 0))
                    comparison.paste(right, (width, 0))
                    left.close()
                    right.close()

                    draw = ImageDraw.Draw(comparison)
                    draw.rectangle(
                        (0, 0, 150, 24), fill=(0, 0, 0)
                    )
                    draw.rectangle(
                        (width, 0, width + 175, 24),
                        fill=(0, 0, 0),
                    )
                    draw.text(
                        (8, 6),
                        "Semantic Input",
                        fill=(255, 255, 255),
                    )
                    draw.text(
                        (width + 8, 6),
                        "Generated Output",
                        fill=(255, 255, 255),
                    )

                    frame = av.VideoFrame.from_image(comparison)
                    comparison.close()

                    for packet in stream.encode(frame):
                        container.mux(packet)

                for packet in stream.encode():
                    container.mux(packet)
            finally:
                container.close()

            target = output_dir / "comparison.mp4"
            temporary.replace(target)

            report.update(
                status="complete",
                codec=codec,
                comparison_mp4=str(target.resolve()),
            )
            report_path.write_text(
                json.dumps(report, indent=2) + "\n"
            )
            logging.info(
                "H3_ARCHIVE_DONE session=%s frames=%s mp4=%s",
                session.id,
                count,
                target,
            )
        except BaseException as error:
            if temporary.exists():
                temporary.unlink()
            report.update(status="error", error=str(error))
            report_path.write_text(
                json.dumps(report, indent=2) + "\n"
            )
            logging.exception(
                "H3_ARCHIVE_ERROR session=%s", session.id
            )

    def finish(self, session):
        self.executor.submit(
            self._finalize,
            session,
            tuple(session.archive_futures),
        )

    def shutdown(self):
        self.executor.shutdown(wait=True)
