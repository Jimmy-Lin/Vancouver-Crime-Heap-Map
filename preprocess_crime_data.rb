require 'date'
require 'json'
require 'csv'

# I orginally wanted to try out Ruby libraries for machine learning
# However, they were a bit slow so I'll just be using Ruby for data preparation
# The heavy linear algebra will be written in Julia for convenience

######################################
## DATA CLEANING / PREPROCESSING 
######################################

seen = {}
frequencies = {}
labels = []

puts "Reading data"

CSV.foreach("crime_csv_all_years.csv") do |row|
  key = row.to_s.to_sym
  # Format : TYPE,YEAR,MONTH,DAY,HOUR,MINUTE,HUNDRED_BLOCK,NEIGHBOURHOOD,X,Y
  if labels.empty?
    labels = row
  elsif !row[1].nil? && !row[2].nil? && !row[3].nil? && row[8] != "0" && row[9] != "0" && seen[key].nil?
  # elsif !row[4].nil? && !row[5].nil? && row[8] != "0" && row[9] != "0" && seen[key].nil?  
    seen[key] = true
    # We will ignore data entries that are insufficient rather than assume defaults
    formatted_time = Time.new(row[1], row[2], row[3])
    # puts formatted_time.to_i
    time = formatted_time.to_time.to_i
    x = row[8].to_f
    y = row[9].to_f
    coord = [time,x,y]
    if frequencies[coord]
      frequencies[coord] += 1
    else
      frequencies[coord] = 1
    end
  end
end

puts "Discretizing data"
hist = frequencies.entries.map{|k,v| k + [v]}
columns = (0..2).map{|i| hist.map{|row| row[i]}}
max = columns.map{|c| c.max}
min = columns.map{|c| c.min}

# Bin size
bin_size = 0.01
# Clear frequencies
frequencies = {}
# Discretize into bins
scaled = hist.map do |row|
  coord = (0..2).map{|i| (row[i]-min[i]).to_f/(max[i]-min[i]) }
  count = row[3]
  key = coord.map{|e| e.round(-Math.log(bin_size, 10).round) }
  if frequencies[key]
    frequencies[key] += count
  else
    frequencies[key] = count
  end
  scaled_row = coord + [count]
end

puts "Computing histogram"

discrete_hist = (0..(1/bin_size).to_i)
  .map{|e| (e*bin_size).round(-Math.log(bin_size, 10).round)}
  .repeated_permutation(3)
  .map do |coord|
    if frequencies[coord]
      coord + [frequencies[coord]]
    else
      coord + [0]
    end
end

count_range = discrete_hist.map{|t,x,y,c| c}.uniq
min_count = count_range.min
max_count = count_range.max
puts "Scaling"
discrete_hist.map!{|t,x,y,c| [t,x,y, ((c-min_count).to_f/(max_count-min_count)).round(-Math.log(bin_size, 10).round)  ]}

puts "Storing histogram"

CSV.open("discrete_hist.csv", "wb") do |csv|
  discrete_hist.each do |row|
    csv << row
  end
end

puts "Density: #{discrete_hist.select{|t,x,y,c| c > 0}.count.to_f/discrete_hist.count}"