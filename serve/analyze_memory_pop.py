import re
from collections import defaultdict

# 读取日志文件
with open('logs/infinisst_api_37391.log', 'r', encoding='utf-8') as f:
    content = f.read()

# 提取所有MEMORY-POP记录
pattern = r'\[MEMORY-POP\] pagetable类型: (\w+), session_id: ([^,]+), 释放页面: (\d+) 个'
matches = re.findall(pattern, content)

# 统计数据
stats = defaultdict(lambda: defaultdict(int))
session_stats = defaultdict(int)
pagetable_stats = defaultdict(int)
total_pages = defaultdict(lambda: defaultdict(int))

for pagetable_type, session_id, pages in matches:
    # 按session和pagetable类型统计次数
    stats[session_id][pagetable_type] += 1
    # 按session统计总次数
    session_stats[session_id] += 1
    # 按pagetable类型统计总次数
    pagetable_stats[pagetable_type] += 1
    # 统计释放的页面总数
    total_pages[session_id][pagetable_type] += int(pages)

print("=" * 80)
print("📊 MEMORY-POP 统计报告")
print("=" * 80)

print(f"\n📈 总体统计:")
print(f"   - 总释放次数: {len(matches)}")
print(f"   - 涉及session数量: {len(stats)}")
print(f"   - pagetable类型: {list(pagetable_stats.keys())}")

print(f"\n📋 按pagetable类型统计:")
for ptype, count in sorted(pagetable_stats.items()):
    total_pages_for_type = sum(total_pages[session_id].get(ptype, 0) for session_id in total_pages.keys())
    print(f"   - {ptype}: {count} 次释放, 总计 {total_pages_for_type} 页")

print(f"\n🔍 按session统计 (前20个最活跃的session):")
sorted_sessions = sorted(session_stats.items(), key=lambda x: x[1], reverse=True)
for i, (session_id, total_count) in enumerate(sorted_sessions[:20]):
    print(f"   {i+1:2d}. {session_id}")
    print(f"       总释放次数: {total_count}")
    for ptype, count in stats[session_id].items():
        pages = total_pages[session_id][ptype]
        print(f"       - {ptype}: {count} 次, {pages} 页")
    print()

print(f"\n📊 详细统计表:")
print(f"{'Session ID':<60} {'Decode释放次数':<12} {'Decode页数':<10} {'Prefill释放次数':<15} {'Prefill页数':<12}")
print("-" * 130)
for session_id in sorted(stats.keys()):
    decode_count = stats[session_id].get('decode', 0)
    decode_pages = total_pages[session_id].get('decode', 0)
    prefill_count = stats[session_id].get('prefill', 0)
    prefill_pages = total_pages[session_id].get('prefill', 0)
    print(f"{session_id:<60} {decode_count:<12} {decode_pages:<10} {prefill_count:<15} {prefill_pages:<12}")

