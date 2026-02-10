import json
import csv
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import logging
import argparse

# Setup default paths
DEFAULT_EXTRACTED_DIR = Path("VallE/egs/sap/extracted")
DEFAULT_OUTPUT_DIR = Path("sap_scripts/sap_data_analysis/prompts")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Extract prompts by category from SAP dataset")
    
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path.cwd(),
        help="The base working dir. Eg. /scratch/<user>"
    )
    
    return parser.parse_args()

def extract_prompts_from_speaker(json_path):
    """
    Extract all prompts from a speaker's JSON file.
    Returns dict: category -> list of (prompt_text, transcript, speaker_id)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    speaker_id = data.get("Contributor ID", "Unknown")
    etiology = data.get("Etiology", "Unknown")
    
    category_prompts = defaultdict(list)
    
    for file_entry in data.get("Files", []):
        prompt_data = file_entry.get("Prompt", {})
        
        prompt_text = prompt_data.get("Prompt Text", "")
        transcript = prompt_data.get("Transcript", "")
        category = prompt_data.get("Category Description", "Unknown")
        sub_category = prompt_data.get("Sub Category Description", "")
        
        if prompt_text:  # Only add if prompt text exists
            category_prompts[category].append({
                'prompt_text': prompt_text,
                'transcript': transcript,
                'sub_category': sub_category,
                'speaker_id': speaker_id,
                'etiology': etiology
            })
    
    return category_prompts


def analyze_prompts(extracted_dir, dataset_type):
    """
    Analyze all prompts in a dataset and organize by category.
    Returns comprehensive prompt statistics.
    """
    dataset_dir = extracted_dir / dataset_type
    speaker_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    if not speaker_dirs:
        logger.warning(f"No speaker directories found in {dataset_dir}")
        return None
    
    # Structure: category -> prompt_text -> {count, speakers, transcripts}
    category_prompt_data = defaultdict(lambda: defaultdict(lambda: {
        'count': 0,
        'speakers': set(),
        'transcripts': set(),
        'sub_categories': set()
    }))
    
    logger.info(f"Analyzing {dataset_type} prompts...")
    
    for speaker_dir in tqdm(speaker_dirs, desc=f"Processing {dataset_type} speakers"):
        json_files = list(speaker_dir.glob("*.json"))
        
        if not json_files:
            continue
        
        try:
            category_prompts = extract_prompts_from_speaker(json_files[0])
            
            for category, prompts in category_prompts.items():
                for prompt_data in prompts:
                    prompt_text = prompt_data['prompt_text']
                    
                    category_prompt_data[category][prompt_text]['count'] += 1
                    category_prompt_data[category][prompt_text]['speakers'].add(prompt_data['speaker_id'])
                    category_prompt_data[category][prompt_text]['transcripts'].add(prompt_data['transcript'])
                    if prompt_data['sub_category']:
                        category_prompt_data[category][prompt_text]['sub_categories'].add(prompt_data['sub_category'])
        
        except Exception as e:
            logger.error(f"Failed to process {speaker_dir.name}: {e}")
    
    return category_prompt_data


def export_prompts_by_category(category_prompt_data, dataset_type, output_dir):
    """
    Export prompts organized by category to separate CSV files.
    One CSV per category.
    """
    if not category_prompt_data:
        logger.warning(f"No prompt data to export for {dataset_type}")
        return
    
    category_output_dir = output_dir / dataset_type / "by_category"
    category_output_dir.mkdir(parents=True, exist_ok=True)
    
    for category, prompts in category_prompt_data.items():
        # Clean category name for filename
        safe_category = category.replace(" ", "_").replace("/", "_")
        output_path = category_output_dir / f"{safe_category}_{dataset_type}.csv"
        
        # Prepare data for CSV
        rows = []
        for prompt_text, data in sorted(prompts.items()):
            rows.append({
                'Category': category,
                'Prompt_Text': prompt_text,
                'Frequency': data['count'],
                'Num_Speakers': len(data['speakers']),
                'Unique_Transcripts': ' | '.join(sorted(data['transcripts'])),
                'Sub_Categories': ' | '.join(sorted(data['sub_categories'])) if data['sub_categories'] else '',
                'Dataset': dataset_type
            })
        
        # Write to CSV
        if rows:
            fieldnames = ['Category', 'Prompt_Text', 'Frequency', 'Num_Speakers', 
                         'Unique_Transcripts', 'Sub_Categories', 'Dataset']
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"Saved {category} prompts: {output_path.name} ({len(rows)} unique prompts)")


def export_all_prompts_combined(category_prompt_data, dataset_type, output_dir):
    """
    Export all prompts to a single CSV file.
    """
    if not category_prompt_data:
        logger.warning(f"No prompt data to export for {dataset_type}")
        return
    
    output_path = output_dir / dataset_type / f"all_prompts_{dataset_type}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for category, prompts in sorted(category_prompt_data.items()):
        for prompt_text, data in sorted(prompts.items()):
            rows.append({
                'Category': category,
                'Prompt_Text': prompt_text,
                'Frequency': data['count'],
                'Num_Speakers': len(data['speakers']),
                'Unique_Transcripts': ' | '.join(sorted(data['transcripts'])),
                'Sub_Categories': ' | '.join(sorted(data['sub_categories'])) if data['sub_categories'] else '',
                'Dataset': dataset_type
            })
    
    if rows:
        fieldnames = ['Category', 'Prompt_Text', 'Frequency', 'Num_Speakers', 
                     'Unique_Transcripts', 'Sub_Categories', 'Dataset']
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Saved combined prompts: {output_path.name} ({len(rows)} total unique prompts)")


def export_category_summary(category_prompt_data, dataset_type, output_dir):
    """
    Export summary statistics per category.
    """
    if not category_prompt_data:
        return
    
    output_path = output_dir / dataset_type / f"category_summary_{dataset_type}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for category in sorted(category_prompt_data.keys()):
        prompts = category_prompt_data[category]
        total_utterances = sum(data['count'] for data in prompts.values())
        unique_prompts = len(prompts)
        all_speakers = set()
        for data in prompts.values():
            all_speakers.update(data['speakers'])
        
        rows.append({
            'Category': category,
            'Unique_Prompts': unique_prompts,
            'Total_Utterances': total_utterances,
            'Num_Speakers': len(all_speakers),
            'Avg_Utterances_Per_Prompt': round(total_utterances / unique_prompts, 2) if unique_prompts > 0 else 0,
            'Dataset': dataset_type
        })
    
    if rows:
        fieldnames = ['Category', 'Unique_Prompts', 'Total_Utterances', 
                     'Num_Speakers', 'Avg_Utterances_Per_Prompt', 'Dataset']
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Saved category summary: {output_path.name}")


def create_master_prompt_list(dev_data, train_data, output_dir):
    """
    Create a master list of all unique prompts across both datasets.
    """
    master_prompts = defaultdict(lambda: {
        'dev_count': 0,
        'train_count': 0,
        'dev_speakers': set(),
        'train_speakers': set(),
        'categories': set(),
        'transcripts': set()
    })
    
    # Collect from DEV
    if dev_data:
        for category, prompts in dev_data.items():
            for prompt_text, data in prompts.items():
                master_prompts[prompt_text]['dev_count'] += data['count']
                master_prompts[prompt_text]['dev_speakers'].update(data['speakers'])
                master_prompts[prompt_text]['categories'].add(category)
                master_prompts[prompt_text]['transcripts'].update(data['transcripts'])
    
    # Collect from TRAIN
    if train_data:
        for category, prompts in train_data.items():
            for prompt_text, data in prompts.items():
                master_prompts[prompt_text]['train_count'] += data['count']
                master_prompts[prompt_text]['train_speakers'].update(data['speakers'])
                master_prompts[prompt_text]['categories'].add(category)
                master_prompts[prompt_text]['transcripts'].update(data['transcripts'])
    
    # Export master list
    output_path = output_dir / "master_prompt_list.csv"
    
    rows = []
    for prompt_text, data in sorted(master_prompts.items()):
        total_count = data['dev_count'] + data['train_count']
        rows.append({
            'Prompt_Text': prompt_text,
            'Categories': ' | '.join(sorted(data['categories'])),
            'Total_Frequency': total_count,
            'DEV_Frequency': data['dev_count'],
            'TRAIN_Frequency': data['train_count'],
            'DEV_Speakers': len(data['dev_speakers']),
            'TRAIN_Speakers': len(data['train_speakers']),
            'Total_Speakers': len(data['dev_speakers']) + len(data['train_speakers']),
            'Unique_Transcripts': ' | '.join(sorted(data['transcripts']))
        })
    
    if rows:
        fieldnames = ['Prompt_Text', 'Categories', 'Total_Frequency', 'DEV_Frequency', 
                     'TRAIN_Frequency', 'DEV_Speakers', 'TRAIN_Speakers', 
                     'Total_Speakers', 'Unique_Transcripts']
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Saved master prompt list: {output_path.name} ({len(rows)} unique prompts)")


def main():
    """Main processing function."""
    args = get_args()
    
    extracted_dir = args.working_dir / DEFAULT_EXTRACTED_DIR
    output_dir = args.working_dir / DEFAULT_OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Working directory: {args.working_dir}")
    logger.info(f"Extracted directory: {extracted_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Check if extracted_dir exists
    if not extracted_dir.exists():
        logger.error(f"Extracted directory does not exist: {extracted_dir}")
        return

    logger.info("="*60)
    logger.info("Starting prompt extraction and analysis")
    logger.info("="*60)
    
    # Analyze DEV dataset
    dev_data = analyze_prompts(extracted_dir, "DEV")
    if dev_data:
        export_prompts_by_category(dev_data, "DEV", output_dir)
        export_all_prompts_combined(dev_data, "DEV", output_dir)
        export_category_summary(dev_data, "DEV", output_dir)
    
    # Analyze TRAIN dataset
    train_data = analyze_prompts(extracted_dir, "TRAIN")
    if train_data:
        export_prompts_by_category(train_data, "TRAIN", output_dir)
        export_all_prompts_combined(train_data, "TRAIN", output_dir)
        export_category_summary(train_data, "TRAIN", output_dir)
    
    # Create master list combining both datasets
    if dev_data or train_data:
        create_master_prompt_list(dev_data, train_data, output_dir)
    
    logger.info("="*60)
    logger.info(f"Prompt extraction complete. Saved to {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()