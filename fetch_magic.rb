require 'oj'
require 'fileutils'
require 'open-uri'
require 'thread'
require 'progress_bar'
require 'net/http'
require 'tempfile'

# Number of threads for concurrent downloads
thread_count = 2

# Default Cards 447 MB
SCRYFALL_DATA_PATH = "/default-cards/default-cards-20241009214135.json"
# All Cards 2.1 GB
# SCRYFALL_DATA_PATH = "/all-cards/all-cards-20241009092303.json"
SCRYFALL_DATA_HOST = "https://data.scryfall.io"

# Download data file if it does not exist
magic_json_file = SCRYFALL_DATA_PATH.split("/").last
if !File.exist?("datasets/#{magic_json_file}")
  puts "Downloading #{magic_json_file}..."
  uri = URI(SCRYFALL_DATA_HOST + SCRYFALL_DATA_PATH)
  http = Net::HTTP.new(uri.host, uri.port)
  http.use_ssl = true
  
  # Get the total size of the file, fallback if not available
  response = http.head(uri.path)
  total_bytes = 0
  
  # Initialize the progress bar
  progress_bar = ProgressBar.new(:rate)

  File.open("datasets/#{magic_json_file}", 'wb') do |file|
    http.get(uri.path) do |str|
      file.write(str)
      total_bytes += str.size
      # Increment progress bar with actual bytes written
      progress_bar.increment!(str.size)
    end
  end
  puts "\n#{magic_json_file} Download complete!"
end

# Read the JSON file
file = "datasets/#{magic_json_file}"
data = Oj.load(File.read(file))

# Directory where images will be saved
output_dir = 'datasets/tcg_magic'

# Helper method to sanitize file and folder names
def sanitize_filename(name)
  name.gsub(/[^0-9A-Za-z.,\- ]/, '')
end

# Create a thread-safe queue to process downloads
queue = Queue.new

# Add each card with an image to the queue
cards = data[0...10]
puts "Found #{cards.size} cards, downloading images..."
cards.each do |card|
  next unless card['image_uris'] && card['image_uris']['png']
  queue << card
end

# Initialize the progress bar
total_images = queue.size
progress_bar = ProgressBar.new(total_images)

# Define a worker method to download the images
def download_images(queue, output_dir, progress_bar)
  until queue.empty?
    card = queue.pop(true) rescue nil
    next unless card

    # Get card properties
    name = card['name']
    set_name = card['set_name']
    collector_number = card['collector_number']

    # Sanitize folder and file names
    sanitized_name = sanitize_filename(name)
    sanitized_set_name = sanitize_filename(set_name)
    sanitized_collector_number = sanitize_filename(collector_number)

    # Create folder for the name if it doesn't exist
    folder_path = File.join(output_dir, sanitized_name)
    FileUtils.mkdir_p(folder_path)

    # Download the image
    image_url = card['image_uris']['png']
    image_name = "#{sanitized_name}_#{sanitized_collector_number}_#{sanitized_set_name}.png"
    image_path = File.join(folder_path, image_name)

    # Skip if the file already exists
    next if File.exist?(image_path)

    # puts "Downloading #{image_name}..."
    URI.open(image_url) do |image|
      File.open(image_path, 'wb') do |file|
        file.write(image.read)
      end
    end

    # Increment the progress bar after each download
    progress_bar.increment!
  end
end

# Start a pool of threads to download images concurrently
threads = []
thread_count.times do
  threads << Thread.new { download_images(queue, output_dir, progress_bar) }
end

# Wait for all threads to complete
threads.each(&:join)

puts "Images Download complete!"
