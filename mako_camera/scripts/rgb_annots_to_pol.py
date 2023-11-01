"""Конвертировать аннотации, написанные для rgb камеры, на pol снимки.

Rgb снимки имеют размер (h, w, 3), а pol (h, w).
Pol снимок разбивается на (h / 2, w / 2, 4),
из-за чего координаты аннотаций необходимо уменьшить в 2 раза.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import shutil


def main(
    rgb_annot_pth: Path, pol_images_dir: Path, pol_cvat_pth: Path,
    copy_images: bool = True
):

    cvat_annots_pth = pol_cvat_pth / 'annotations.xml'
    cvat_images_dir = pol_cvat_pth / 'images'
    cvat_images_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(rgb_annot_pth)
    root = tree.getroot()

    images_tags = root.findall('image')
    for image_tag in images_tags:
        pol_img_name = 'pol_' + image_tag.get('name')[4:-4] + '.npy'

        if copy_images:
            src_pth = pol_images_dir / pol_img_name
            dst_pth = cvat_images_dir / pol_img_name
            # Иногда индексы rgb и pol фреймов отличаются на 1
            try:
                shutil.copy2(src_pth, dst_pth)
            except FileNotFoundError:
                src_name = src_pth.name
                src_parent = src_pth.parent
                dst_name = dst_pth.name
                dst_parent = dst_pth.parent
                print(f'Could not find {src_name}.')

                index = int(src_name[4:-4])
                src_pth = src_parent / (
                    src_name[:4] + str(index - 1) + src_name[-4:])
                dst_pth = dst_parent / (
                    dst_name[:4] + str(index - 1) + dst_name[-4:])
                print(f'Trying {src_pth.name}')
                try:
                    shutil.copy2(src_pth, dst_pth)
                except FileNotFoundError:

                    src_pth = src_parent / (
                        src_name[:4] + str(index + 1) + src_name[-4:])
                    dst_pth = dst_parent / (
                        dst_name[:4] + str(index + 1) + dst_name[-4:])
                    print(f'Trying {src_pth.name}')
                    try:
                        shutil.copy2(src_pth, dst_pth)
                    except FileNotFoundError:
                        print('Unsuccessful')
            finally:
                pol_img_name = src_pth.name

        image_tag.set('name', pol_img_name)
        image_tag.set('width', str(int(image_tag.get('width')) // 2))
        image_tag.set('height', str(int(image_tag.get('height')) // 2))
        
        boxes_tags = image_tag.findall('box')
        for box_tag in boxes_tags:
            x1 = float(box_tag.get('xtl'))
            y1 = float(box_tag.get('ytl'))
            x2 = float(box_tag.get('xbr'))
            y2 = float(box_tag.get('ybr'))
            box_tag.set('xtl', str(x1 / 2))
            box_tag.set('ytl', str(y1 / 2))
            box_tag.set('xbr', str(x2 / 2))
            box_tag.set('ybr', str(y2 / 2))
    tree.write(cvat_annots_pth, encoding='utf-8', xml_declaration=True)
    

if __name__ == '__main__':
    rgb_annot_pth = Path('data/tank_3set_rgb/annotations.xml')
    pol_cvat_pth = Path('data/tank_3set_pol')
    pol_images_dir = Path('data/camera/2023_09_26_tank3/raw_pol')
    main(rgb_annot_pth, pol_images_dir, pol_cvat_pth)
