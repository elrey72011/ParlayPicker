import requests

# ✅ Base URL is correct for V2
base_url = "https://api.elections.kalshi.com/trade-api/v2"

# Get all markets for the KXHIGHNY series
markets_url = f"{base_url}/markets"
params = {
    'series_ticker': 'KXHIGHNY',
    'status': 'open'
}

response = requests.get(markets_url, params=params)
markets_data = response.json()

print(f"\nActive markets in KXHIGHNY series:")

if 'markets' in markets_data:
    for market in markets_data['markets']:
        # ⚠️ FIX: Use 'yes_bid' or 'last_price' instead of 'yes_price'
        price = market.get('yes_bid', 0)  # Default to 0 if empty
        volume = market.get('volume', 0)
        
        print(f"- {market['ticker']}: {market['title']}")
        print(f"  Event: {market['event_ticker']}")
        print(f"  Yes Bid: {price}¢ | Volume: {volume}")
        print()

    # Get details for a specific event
    if markets_data['markets']:
        event_ticker = markets_data['markets'][0]['event_ticker']
        event_url = f"{base_url}/events/{event_ticker}"
        
        event_response = requests.get(event_url)
        event_data = event_response.json()

        print(f"Event Details:")
        # 'event' key contains the metadata
        if 'event' in event_data:
            print(f"Title: {event_data['event']['title']}")
            print(f"Category: {event_data['event']['category']}")
else:
    print("No markets found or API error:", markets_data)
