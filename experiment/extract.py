def extract_valid_lines(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            if line.startswith('prefilling InputMetadata') and 'num_prompts=' in line:
                num_prompts_part = line.split('num_prompts=')[1].split(',')[0]
                if num_prompts_part.isdigit() and int(num_prompts_part) != 0:
                    output_file.write(line)

input_file_path = 'separate_baichuang.txt'  # Update with the actual file path
output_file_path = 'separate_baichuang_prefill.txt'  # Update with the desired output file path

extract_valid_lines(input_file_path, output_file_path)