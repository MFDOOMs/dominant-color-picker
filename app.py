from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning


NUMBER_OF_COLORS = 5
MAX_IMAGE_SIZE = (420, 420)


@dataclass(frozen=True)
class DominantColor:
    rgb: tuple[int, int, int]
    percentage: float
    pixel_count: int

    @property
    def hex_code(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(*self.rgb)


def resize_for_analysis(image: Image.Image) -> Image.Image:
    resized_image = image.convert("RGB")
    resized_image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

    if resized_image.width * resized_image.height < NUMBER_OF_COLORS:
        resized_image = resized_image.resize(
            (NUMBER_OF_COLORS, NUMBER_OF_COLORS), Image.Resampling.NEAREST
        )

    return resized_image


@st.cache_data(show_spinner=False)
def find_dominant_colors(
    image_bytes: bytes,
) -> tuple[tuple[DominantColor, ...], tuple[int, int], int, int]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        analysis_image = resize_for_analysis(image)

    pixels = np.asarray(analysis_image, dtype=np.float64).reshape(-1, 3)
    unique_color_count = len(np.unique(pixels, axis=0))

    model = KMeans(n_clusters=NUMBER_OF_COLORS, random_state=42, n_init=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        labels = model.fit_predict(pixels)

    centers = np.clip(np.rint(model.cluster_centers_), 0, 255).astype(int)
    cluster_sizes = np.bincount(labels, minlength=NUMBER_OF_COLORS)
    sorted_indices = np.argsort(cluster_sizes)[::-1]
    total_pixels = pixels.shape[0]

    colors = tuple(
        DominantColor(
            rgb=tuple(int(value) for value in centers[index]),
            percentage=float(cluster_sizes[index] / total_pixels * 100),
            pixel_count=int(cluster_sizes[index]),
        )
        for index in sorted_indices
    )

    return colors, analysis_image.size, total_pixels, unique_color_count


def load_preview_image(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def text_color_for_background(rgb: tuple[int, int, int]) -> str:
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "#0F172A" if luminance > 170 else "#FFFFFF"


def create_palette_report(colors: tuple[DominantColor, ...]) -> str:
    report_lines = ["Dominant Color Picker - Hasil Analisis K-Means", ""]
    for rank, color in enumerate(colors, start=1):
        report_lines.append(
            f"{rank}. {color.hex_code} | RGB{color.rgb} | {color.percentage:.2f}%"
        )
    return "\n".join(report_lines)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --ink: #0f172a;
                --muted: #475569;
                --panel: rgba(255, 255, 255, 0.86);
                --border: rgba(148, 163, 184, 0.22);
            }
            .stApp {
                background:
                    radial-gradient(circle at 8% 4%, rgba(56, 189, 248, 0.15), transparent 30%),
                    radial-gradient(circle at 92% 3%, rgba(139, 92, 246, 0.15), transparent 28%),
                    #f6f8fc;
            }
            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            [data-testid="stToolbarActions"],
            [data-testid="stDecoration"],
            #MainMenu {
                display: none;
            }
            .block-container {
                max-width: 1120px;
                padding-top: 2.4rem;
                padding-bottom: 2rem;
            }
            [data-testid="stMarkdownContainer"],
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li,
            [data-testid="stMarkdownContainer"] h2,
            [data-testid="stMarkdownContainer"] h3,
            [data-testid="stMarkdownContainer"] h4 {
                color: var(--ink);
            }
            .hero {
                padding: 2.15rem 2.25rem;
                border-radius: 26px;
                margin-bottom: 1.45rem;
                color: #ffffff;
                background: linear-gradient(120deg, #102340, #244a8d 58%, #693ab4);
                box-shadow: 0 18px 46px rgba(24, 48, 91, 0.20);
            }
            .hero-tag {
                display: inline-flex;
                border: 1px solid rgba(255, 255, 255, 0.25);
                border-radius: 99px;
                padding: 0.32rem 0.85rem;
                margin-bottom: 0.85rem;
                font-size: 0.78rem;
                letter-spacing: 0.09em;
                text-transform: uppercase;
            }
            .hero h1 {
                font-size: clamp(2rem, 4vw, 2.8rem);
                line-height: 1.1;
                margin: 0 0 0.55rem;
            }
            .hero p {
                max-width: 700px;
                color: rgba(255, 255, 255, 0.80);
                font-size: 1rem;
                margin: 0;
            }
            .info-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 1rem;
                margin: 0 0 1.45rem;
            }
            .info-card {
                padding: 1.1rem 1.2rem;
                border-radius: 18px;
                background: var(--panel);
                border: 1px solid var(--border);
                box-shadow: 0 5px 16px rgba(23, 38, 67, 0.04);
                color: var(--ink);
            }
            .info-card h3 {
                font-size: 1rem;
                margin: 0 0 0.55rem;
            }
            .info-card p,
            .info-card ol {
                color: var(--muted);
                font-size: 0.9rem;
                line-height: 1.62;
                margin: 0;
            }
            .info-card ol {
                padding-left: 1.2rem;
            }
            @media (max-width: 700px) {
                .info-grid {
                    grid-template-columns: 1fr;
                }
            }
            [data-testid="stFileUploader"] {
                border-radius: 18px;
                padding: 0.55rem 0.65rem 0.1rem;
                background: var(--panel);
                border: 1px solid var(--border);
            }
            [data-testid="stImage"] img {
                border-radius: 18px;
                box-shadow: 0 9px 27px rgba(23, 38, 67, 0.08);
            }
            [data-testid="stMetric"] {
                background: var(--panel);
                padding: 0.72rem 0.85rem;
                border: 1px solid var(--border);
                border-radius: 14px;
            }
            [data-testid="stMetricLabel"],
            [data-testid="stMetricLabel"] *,
            [data-testid="stMetricValue"],
            [data-testid="stMetricValue"] * {
                color: var(--ink) !important;
            }
            [data-testid="stMetricLabel"] {
                font-weight: 600;
            }
            [data-testid="stMetricValue"],
            [data-testid="stMetricValue"] * {
                font-size: clamp(1.35rem, 2.3vw, 2.1rem) !important;
                letter-spacing: -0.025em;
                white-space: nowrap;
            }
            [data-testid="stImageCaption"] {
                color: var(--muted) !important;
            }
            [data-testid="stDownloadButton"] button {
                color: #ffffff !important;
                font-weight: 600;
                border: none;
                background: linear-gradient(110deg, #2563eb, #4f46e5);
            }
            [data-testid="stDownloadButton"] button:hover {
                color: #ffffff !important;
                background: linear-gradient(110deg, #1d4ed8, #4338ca);
            }
            [data-testid="stDownloadButton"] button p {
                color: #ffffff !important;
            }
            .upload-empty {
                margin-top: 1.2rem;
                padding: 2.2rem 1rem;
                border: 1px dashed #b8c6dc;
                border-radius: 18px;
                text-align: center;
                color: var(--muted);
                background: rgba(255, 255, 255, 0.58);
            }
            .palette-strip {
                display: flex;
                height: 86px;
                overflow: hidden;
                border-radius: 18px;
                margin: 0.7rem 0 1.2rem;
                box-shadow: 0 9px 24px rgba(23, 38, 67, 0.10);
            }
            .palette-part {
                flex: 1;
            }
            .color-card {
                border-radius: 17px;
                border: 1px solid var(--border);
                background: var(--panel);
                overflow: hidden;
                min-height: 177px;
                box-shadow: 0 5px 16px rgba(23, 38, 67, 0.05);
                margin-bottom: 0.45rem;
            }
            .swatch {
                height: 76px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 700;
                font-size: 0.8rem;
                letter-spacing: 0.08em;
            }
            .color-details {
                padding: 0.72rem 0.72rem 0.66rem;
                color: var(--ink);
            }
            .color-details strong {
                font-size: 1.02rem;
            }
            .color-details span {
                display: block;
                color: var(--muted);
                font-size: 0.8rem;
                line-height: 1.52;
            }
            .copy-tip {
                margin: 0.1rem 0 0.75rem;
                color: var(--muted);
                font-size: 0.88rem;
            }
            .footer {
                text-align: center;
                color: var(--muted);
                font-size: 0.84rem;
                margin-top: 2.35rem;
                padding-top: 1.1rem;
                border-top: 1px solid #dde5f2;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <section class="hero">
            <div class="hero-tag">Image Color Analysis</div>
            <h1>Dominant Color Picker</h1>
            <p>
                Ubah gambar menjadi palet warna dalam hitungan detik.
                Temukan lima warna utama melalui pengelompokan pixel RGB
                menggunakan K-Means Clustering.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_information() -> None:
    st.markdown(
        """
        <section class="info-grid">
            <div class="info-card">
                <h3>Cara menggunakan</h3>
                <ol>
                    <li>Upload gambar JPG, JPEG, atau PNG.</li>
                    <li>Tunggu proses clustering selesai.</li>
                    <li>Lihat dan salin warna HEX dari palet hasil analisis.</li>
                </ol>
            </div>
            <div class="info-card">
                <h3>Proses K-Means</h3>
                <p>
                    Gambar diperkecil lalu pixel RGB dibagi ke dalam
                    <strong>5 cluster</strong>. Centroid setiap cluster menjadi
                    warna dominan dan jumlah pixel menentukan persentasenya.
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_palette(colors: tuple[DominantColor, ...]) -> None:
    palette_html = "".join(
        f'<div class="palette-part" style="background: {color.hex_code};"></div>'
        for color in colors
    )
    st.markdown(
        f'<div class="palette-strip">{palette_html}</div>', unsafe_allow_html=True
    )

    st.markdown("#### Detail warna")
    st.markdown(
        '<div class="copy-tip">Tekan ikon salin pada blok HEX untuk menyalin kode warna.</div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(NUMBER_OF_COLORS)

    for rank, (column, color) in enumerate(zip(columns, colors), start=1):
        foreground = text_color_for_background(color.rgb)
        with column:
            st.markdown(
                f"""
                <div class="color-card">
                    <div class="swatch" style="background: {color.hex_code}; color: {foreground};">
                        WARNA {rank}
                    </div>
                    <div class="color-details">
                        <strong>{color.hex_code}</strong>
                        <span>RGB {color.rgb}</span>
                        <span>{color.percentage:.2f}% dari pixel</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.code(color.hex_code, language="text")


def render_footer() -> None:
    st.markdown(
        '<div class="footer">Dominant Color Picker | Powered by Streamlit and K-Means Clustering</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Dominant Color Picker",
        page_icon=":material/palette:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_custom_css()
    render_header()
    render_information()

    uploaded_file = st.file_uploader(
        "Upload gambar untuk dianalisis",
        type=["jpg", "jpeg", "png"],
        help="Format yang didukung: JPG, JPEG, PNG.",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="upload-empty">
                <strong>Belum ada gambar yang dipilih.</strong><br>
                Upload sebuah gambar untuk menampilkan palet lima warna dominan.
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_footer()
        return

    image_bytes = uploaded_file.getvalue()
    try:
        preview_image = load_preview_image(image_bytes)
    except (UnidentifiedImageError, OSError):
        st.error("File tidak dapat dibaca sebagai gambar JPG atau PNG yang valid.")
        render_footer()
        return

    with st.spinner("Mengelompokkan warna pixel dengan K-Means..."):
        colors, resized_size, sampled_pixels, unique_color_count = find_dominant_colors(
            image_bytes
        )

    image_column, insight_column = st.columns([1.18, 1], gap="large")
    with image_column:
        st.markdown("### Gambar yang diupload")
        st.image(
            preview_image,
            caption=uploaded_file.name,
            use_container_width=True,
        )

    with insight_column:
        st.markdown("### Ringkasan analisis")
        width, height = resized_size
        metric_a, metric_b = st.columns(2)
        metric_a.metric("Ukuran analisis", f"{width} x {height} px")
        metric_b.metric("Pixel sampel", f"{sampled_pixels:,}")
        metric_c, metric_d = st.columns(2)
        metric_c.metric("Jumlah cluster", NUMBER_OF_COLORS)
        metric_d.metric("Warna teratas", colors[0].hex_code)

        st.markdown(
            """
            Setiap pixel RGB diperlakukan sebagai titik data tiga dimensi.
            Centroid hasil K-Means menjadi representasi warna dominan, sedangkan
            ukuran cluster menentukan persentase kemunculannya.
            """
        )
        st.download_button(
            "Download hasil palet (.txt)",
            data=create_palette_report(colors),
            file_name="dominant_color_palette.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("### Palet warna dominan")
    if unique_color_count < NUMBER_OF_COLORS:
        st.info(
            "Gambar hasil resize memiliki kurang dari lima warna RGB unik. "
            "Karena jumlah cluster tetap 5, beberapa centroid dapat sama atau "
            "memiliki persentase 0%."
        )
    render_palette(colors)

    with st.expander("Penjelasan singkat algoritma K-Means"):
        st.markdown(
            """
            1. Gambar diubah menjadi mode **RGB** lalu diperkecil maksimum
               **420 x 420 pixel** agar komputasi lebih ringan.
            2. Array gambar dibentuk menjadi daftar pixel dengan fitur
               `(Red, Green, Blue)`.
            3. `KMeans(n_clusters=5, random_state=42, n_init=10)` mencari lima
               centroid warna secara konsisten.
            4. Label cluster dihitung untuk memperoleh persentase pixel pada
               setiap centroid.
            """
        )

    render_footer()


if __name__ == "__main__":
    main()
