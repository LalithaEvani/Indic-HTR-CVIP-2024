import argparse
import base64
from html import escape
from io import BytesIO
from pathlib import Path
from tqdm import tqdm

from PIL import Image

# Import necessary classes and functions from the provided code
from strhub.data.module import LmdbDataset


def main():
    parser = argparse.ArgumentParser(description="Generate HTML file containing images from LMDB dataset")
    parser.add_argument('lmdb_path', type=str, help="Path to the LMDB dataset")
    parser.add_argument('--max_label_len', type=int, default=100, help="Maximum length of text labels")
    parser.add_argument('--min_image_dim', type=int, default=0, help="Minimum image dimension")
    parser.add_argument('--remove_whitespace', action='store_true', help="Remove whitespace from labels")
    parser.add_argument('--normalize_unicode', action='store_true', help="Normalize Unicode characters in labels")
    parser.add_argument('--unlabelled', action='store_true', help="Treat the dataset as unlabelled")
    args = parser.parse_args()

    charset = " ਃ ਂ ੂ ੍ ੱ ੋ ੵ ੈ ਾ ਿ ੑ ੌ ੰ ੁ ਁ ੇ ੀ ਼ੜਬਮਊਥ੩ਣਅਜਟਆਝੳਞਏਓਠਇ੯੦ਦਗਛਙਵਲਕ੬ਚ੧ਔਹ੨ਐੴਖਫਧ੪ਸਨਈਡ੫ੲ੭੮ਢਉਸ਼ਭਰਪਯਘਤਖ਼ਲ਼ਜ਼ਗ਼ਫ਼\"#$%&'()*+,-./:;<=>?@[\\]^_{|}~0123456789 "
    max_label_len= 35
    remove_whitespace = True
    normalize_unicode = False
    unlabelled = False

    # Create LMDB dataset
    dataset = LmdbDataset(args.lmdb_path, charset, max_label_len,
                          args.min_image_dim, remove_whitespace,
                          normalize_unicode, unlabelled)

    # Generate HTML content
    html_content = generate_html(dataset)

    # Save HTML content to file
    save_html(html_content,'val_images.html')


def generate_html(dataset):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            table {
                font-family: Arial, sans-serif;
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid #dddddd;
                text-align: left;
                padding: 8px;
            }
            th {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>

    <h2>Images</h2>
    <table>
      <tr>
        <th>Image</th>
        <th>Label</th>
      </tr>
    """

    for index in tqdm(range(1000)):
        img, label = dataset[index]
        img_base64 = image_to_base64(img)
        html_content += f"""
        <tr>
            <td><img src="data:image/png;base64,{img_base64}" alt="Image" width="100"></td>
            <td>{escape(str(label))}</td>
        </tr>
        """

    html_content += """
    </table>

    </body>
    </html>
    """
    return html_content


def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def save_html(html_content, output_html):
    with open(output_html, 'w') as f:
        f.write(html_content)


if __name__ == '__main__':
    main()

