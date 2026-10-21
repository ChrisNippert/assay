package server

import (
	"time"

	"ollamui/internal/chats"
)

// ChatGroup is one dated section of the sidebar chat list.
type ChatGroup struct {
	Label string
	Chats []chats.Summary
}

// ChatGroups buckets the page's chats into Today / Yesterday / Previous 7 days / Older
// (by local calendar day), keeping order and omitting empty groups.
// Chats in a project are listed under the project instead.
func (p *page) ChatGroups() []ChatGroup { return groupChats(p.looseChats(), time.Now()) }

func groupChats(list []chats.Summary, now time.Time) []ChatGroup {
	y, m, d := now.Date()
	today := time.Date(y, m, d, 0, 0, 0, 0, now.Location())
	labels := []string{"Today", "Yesterday", "Previous 7 days", "Older"}
	buckets := make([][]chats.Summary, len(labels))
	for _, c := range list {
		u := c.Updated.In(now.Location())
		i := 3
		switch {
		case !u.Before(today):
			i = 0
		case !u.Before(today.AddDate(0, 0, -1)):
			i = 1
		case !u.Before(today.AddDate(0, 0, -7)):
			i = 2
		}
		buckets[i] = append(buckets[i], c)
	}
	var out []ChatGroup
	for i, b := range buckets {
		if len(b) > 0 {
			out = append(out, ChatGroup{Label: labels[i], Chats: b})
		}
	}
	return out
}
