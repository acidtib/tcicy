require 'oj'
require 'fileutils'
require 'open-uri'
require 'thread'
require 'progress_bar'
require 'net/http'
require 'tempfile'
require 'etc'

# Number of threads for concurrent downloads
thread_count = Etc.nprocessors
max_cards_to_download = 1000 # Set to nil if you want to download all cards


puts "Using #{thread_count} threads"

scryfall_data_paths = [
  # 203 MB
  "/unique-artwork/unique-artwork-20241010211157.json",

  # 146 MB
  "/oracle-cards/oracle-cards-20241010210413.json",

  # 447 MB
  "/default-cards/default-cards-20241010212018.json",

  # 2.1 GB
  # "/all-cards/all-cards-20241010213949.json"
]
scryfall_data_host = "https://data.scryfall.io"

# Directory where images will be saved
output_dir = 'datasets/tcg_magic'

# Helper method to sanitize file and folder names
def sanitize_filename(name)
  name.gsub(/[^\w]/, '_')
end

# Create a thread-safe queue to process downloads
queue = Queue.new

scryfall_data_paths.each do |data_path|
  magic_json_file = data_path.split("/").last
  file_path = "datasets/#{magic_json_file}"

  # Download data file if it does not exist
  unless File.exist?(file_path)
    puts "Downloading #{magic_json_file}..."
    uri = URI(scryfall_data_host + data_path)
    response = Net::HTTP.get(uri)
    File.write(file_path, response)
    puts "\n#{magic_json_file} Download complete!"
  else
    puts "#{magic_json_file} already exists. Skipping download."
  end

  # Read the JSON file
  data = Oj.load(File.read(file_path))
  
  cards = max_cards_to_download.nil? ? data : data[0...max_cards_to_download]

  # Add each card with an image to the queue
  cards.each do |card|
    next unless card['image_uris'] && card['image_uris']['png']
    queue << card
  end
end

# Initialize the progress bar
total_images = queue.size
progress_bar = ProgressBar.new(total_images)
progress_mutex = Mutex.new

puts "Found #{total_images} cards, downloading images..."

# Worker method to download the images
def download_image(image_url, image_path)
  begin
    URI.open(image_url) do |image|
      File.open(image_path, 'wb') do |file|
        file.write(image.read)
      end
    end
  rescue => e
    puts "Error downloading #{image_url}: #{e.message}"
    sleep(1) # Small delay before retry
    retry
  end
end

def download_images(queue, output_dir, progress_bar, progress_mutex)
  until queue.empty?
    card = queue.pop(true) rescue nil
    next unless card

    name = card['name']
    set_name = card['set_name']
    collector_number = card['collector_number']
    sanitized_name = sanitize_filename(name)
    sanitized_set_name = sanitize_filename(set_name)
    sanitized_collector_number = sanitize_filename(collector_number)
    folder_path = File.join(output_dir, sanitized_name)
    FileUtils.mkdir_p(folder_path)
    image_url = card['image_uris']['png']
    image_name = "#{sanitized_name}_#{sanitized_collector_number}_#{sanitized_set_name}.png"
    image_path = File.join(folder_path, image_name)

    if File.exist?(image_path)
      progress_mutex.synchronize { progress_bar.increment! }
      next
    end

    download_image(image_url, image_path)

    # Increment the progress bar safely
    progress_mutex.synchronize { progress_bar.increment! }
  end
end

# Start a pool of threads to download images concurrently
threads = []
thread_count.times do
  threads << Thread.new { download_images(queue, output_dir, progress_bar, progress_mutex) }
end

# Wait for all threads to complete
threads.each(&:join)

puts "Images download complete!"
