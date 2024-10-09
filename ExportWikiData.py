import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_education_and_awards(scientist_url):
    response = requests.get(scientist_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the education paragraph
    education = soup.find('span', {'id': 'Education'})
    if education:
        education_paragraph = education.find_next('p')
    else:
        education_paragraph = None

    # Find the awards section
    awards = soup.find('span', {'id': 'Awards_and_honors'}) or soup.find('span', {'id': 'Awards'}) or soup.find('span', {'id': 'Honors_&_Awards'}) or soup.find('span', {'id': 'Honours_and_awards'})  or soup.find('span', {'id': 'Awards_and_recognition'})   or soup.find('span', {'id': 'Honors_and_awards'}) or soup.find('span', {'id': 'Awards_and_honors_list'}) or soup.find('span', {'id': 'Awards_and_honours'})
    if awards:
        awards_paragraph = awards.find_next('ul')
        awards_list = awards_paragraph.find_all('li')
        num_awards = len(awards_list)
    else:
        num_awards = 0  # Set num_awards to 0 if no awards are found

    return education_paragraph.text.strip() if education_paragraph else None, num_awards

def main_part1():
    url = "https://en.wikipedia.org/wiki/List_of_computer_scientists"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Initialize lists to store data
    scientists_data = []

    # Find all sections with names
    sections = soup.find_all('span', {'class': 'mw-headline'})

    # Counter to limit to the first 100 scientists
    count = 0

    for section in sections:
        current_element = section.find_next('li')

        while current_element and current_element.find('a') and count < 683:
            name = current_element.find('a').text.strip()
            scientist_url = "https://en.wikipedia.org" + current_element.find('a')['href']

            # Get education and awards information for the current scientist
            education, num_awards = get_education_and_awards(scientist_url)

            # Append data to the list
            scientists_data.append({'Surname': name, '#Awards': num_awards, 'Education': education,})

            current_element = current_element.find_next('li')
            count += 1

    # Create a DataFrame using pandas
    df = pd.DataFrame(scientists_data)

    # Save the DataFrame to a CSV file
    df.to_csv('Dataset.csv', index=False)

def main_part2():
    def search_dblp(name):
        base_url = "https://dblp.org/search?q=" + name.replace(" ", "+")
        response = requests.get(base_url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            matches_element = soup.find(id="completesearch-info-matches")

            if matches_element:
                matches_text = matches_element.get_text(strip=True)
                # Extract the number from the "found X matches" text
                if "found one match" in matches_text:
                    return "1"
                elif "found 1 match" in matches_text:
                    return "1"
                elif "no matches" in matches_text:
                    return "0"
                else:
                    matches_count = matches_text.split()[1]
                    return matches_count

        return "0"

    # Read the CSV file
    csv_file_path = "Dataset.csv"
    df = pd.read_csv(csv_file_path)

    # Apply the search_dblp function to each row in the "Name" column and create a new column "DBLP_Record"
    df["#DBLP_Record"] = df["Surname"].apply(search_dblp)

    # Save the updated DataFrame to the same CSV file
    df.to_csv(csv_file_path, index=False)

    # Print the updated DataFrame
    print(df)

if __name__ == "__main__":
    main_part1()
    main_part2()
