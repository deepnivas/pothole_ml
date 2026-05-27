import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

def main():
    annotations_dir = r"C:\Users\E028.28\Downloads\archive (2)\annotations"
    output_csv = r"C:\Users\E028.28\Downloads\files\data\telemetry.csv"
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    print("Starting dataset parsing and telemetry generation...")
    
    rows = []
    rng = np.random.default_rng(42)
    
    # We have files potholes0.xml to potholes664.xml
    total_files = 665
    missing_count = 0
    
    for i in range(total_files):
        xml_filename = f"potholes{i}.xml"
        image_filename = f"potholes{i}.png"
        xml_path = os.path.join(annotations_dir, xml_filename)
        
        if not os.path.exists(xml_path):
            missing_count += 1
            # Fallback values if XML is missing
            num_potholes = 0
            total_area_fraction = 0.0
        else:
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Get image dimensions
                size_node = root.find("size")
                if size_node is not None:
                    width = float(size_node.find("width").text)
                    height = float(size_node.find("height").text)
                else:
                    width, height = 450.0, 300.0  # default
                
                num_potholes = 0
                total_area_fraction = 0.0
                
                for obj in root.findall("object"):
                    if obj.find("name").text == "pothole":
                        num_potholes += 1
                        bndbox = obj.find("bndbox")
                        if bndbox is not None:
                            xmin = float(bndbox.find("xmin").text)
                            ymin = float(bndbox.find("ymin").text)
                            xmax = float(bndbox.find("xmax").text)
                            ymax = float(bndbox.find("ymax").text)
                            
                            area = (xmax - xmin) * (ymax - ymin)
                            total_area_fraction += area / (width * height)
            except Exception as e:
                print(f"Error parsing {xml_filename}: {e}")
                num_potholes = 0
                total_area_fraction = 0.0
        
        # Calculate a realistic severity label [1.0, 10.0] based on visual severity
        # More potholes and larger size = higher severity
        severity = 1.0 + 1.5 * num_potholes + 6.0 * total_area_fraction
        severity = np.clip(severity, 1.0, 10.0)
        
        # Generate severity-correlated synthetic IMU telemetry values
        # Z-acceleration (az) gets a strong peak for higher severity potholes
        ts = 1718000000 + i * 100
        ax = rng.normal(0.0, 0.15)
        ay = rng.normal(0.0, 0.15)
        # 9.8 (gravity) + severity-dependent impact spike + high-freq noise
        az_spike = 9.8 + (severity - 1.0) * 1.5 + rng.normal(0.0, 0.25)
        gx = rng.normal(0.0, 0.04)
        gy = rng.normal(0.0, 0.04)
        gz = rng.normal(0.0, 0.04)
        
        rows.append({
            "timestamp": ts,
            "ax": round(ax, 4),
            "ay": round(ay, 4),
            "az": round(az_spike, 4),
            "gx": round(gx, 4),
            "gy": round(gy, 4),
            "gz": round(gz, 4),
            "severity_label": round(float(severity), 2),
            "image_file": image_filename,
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    
    print(f"Dataset preparation complete.")
    print(f"Total samples: {len(df)}")
    print(f"Telemetry CSV saved to: {output_csv}")
    print(f"Missing XML files fallback count: {missing_count}")
    print(f"Severity stats:\n{df['severity_label'].describe()}")

if __name__ == "__main__":
    main()
